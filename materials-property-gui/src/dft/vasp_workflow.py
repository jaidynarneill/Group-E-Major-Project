import re
import shlex
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

try:
    from analysis.elastic_properties import (
        EV_A3_TO_GPA,
        REFERENCE_LATTICE_PARAMETERS,
        STRAINS,
        _normalize_material,
        _normalize_structure,
        birch_murnaghan,
        derive_cubic_properties,
        fit_linear_response,
    )
    from analysis.surface_energy import SURFACE_ORIENTATIONS
    from dft.remote_vasp import (
        MODULEFILE_DIRECTORY,
        REMOTE_SCRATCH_DIRECTORY,
        SSH_HOST,
        VASP_PREREQUISITE_MODULES,
        VASP_MODULE,
    )
except ModuleNotFoundError as error:
    if error.name not in {"analysis", "dft"}:
        raise
    from src.analysis.elastic_properties import (
        EV_A3_TO_GPA,
        REFERENCE_LATTICE_PARAMETERS,
        STRAINS,
        _normalize_material,
        _normalize_structure,
        birch_murnaghan,
        derive_cubic_properties,
        fit_linear_response,
    )
    from src.analysis.surface_energy import SURFACE_ORIENTATIONS
    from src.dft.remote_vasp import (
        MODULEFILE_DIRECTORY,
        REMOTE_SCRATCH_DIRECTORY,
        SSH_HOST,
        VASP_PREREQUISITE_MODULES,
        VASP_MODULE,
    )


VASP_WORKFLOW_VERSION = 1
VASP_KPOINTS_BULK = (8, 8, 8)
VASP_KPOINTS_SURFACE = (8, 8, 1)
VASP_ENCUT_FACTOR = 1.3
VASP_SURFACE_LAYERS = 8
VASP_SURFACE_VACUUM = 10.0
VASP_SURFACE_FMAX = 0.05
VASP_SURFACE_MAX_STEPS = 200
VASP_CACHE_BACKEND = (
    "VASP",
    VASP_MODULE,
    VASP_WORKFLOW_VERSION,
    VASP_KPOINTS_BULK,
    VASP_KPOINTS_SURFACE,
    VASP_ENCUT_FACTOR,
    VASP_SURFACE_LAYERS,
    VASP_SURFACE_VACUUM,
    VASP_SURFACE_FMAX,
    VASP_SURFACE_MAX_STEPS,
)


def _run_script_path():
    if getattr(sys, "frozen", False):
        bundle_directory = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        return bundle_directory / "src" / "dft" / "run.sh"
    return Path(__file__).with_name("run.sh")


def _write_vasp_task(directory, atoms, *, relax=False, kpoints=None):
    from ase.io import write

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / "POSCAR", atoms, format="vasp", direct=True, vasp5=True, sort=True)
    mesh = kpoints or VASP_KPOINTS_BULK
    (directory / "KPOINTS").write_text(
        "Automatic mesh\n0\nGamma\n" + " ".join(map(str, mesh)) + "\n0 0 0\n",
        encoding="ascii",
    )
    incar = [
        "PREC = Accurate",
        "ENCUT = @ENCUT@",
        "EDIFF = 1E-8",
        "ISMEAR = 0",
        "SIGMA = 0.05",
        "LREAL = .FALSE.",
        "LASPH = .TRUE.",
        "ISYM = 0",
        "ISIF = 2",
        f"IBRION = {2 if relax else -1}",
        f"NSW = {VASP_SURFACE_MAX_STEPS if relax else 0}",
        "LWAVE = .FALSE.",
        "LCHARG = .FALSE.",
    ]
    if relax:
        incar.append(f"EDIFFG = -{VASP_SURFACE_FMAX}")
    (directory / "INCAR.template").write_text("\n".join(incar) + "\n", encoding="ascii")


def build_eos_tasks(material, lattice_structure, directory):
    from ase.build import bulk

    material_name, symbol = _normalize_material(material)
    crystal_structure = _normalize_structure(lattice_structure)
    reference = REFERENCE_LATTICE_PARAMETERS[symbol]
    lattice_parameters = np.linspace(reference * 0.97, reference * 1.03, 13)
    tasks = []
    for index, lattice_parameter in enumerate(lattice_parameters):
        atoms = bulk(symbol, crystal_structure, a=float(lattice_parameter), cubic=True)
        name = f"eos_{index:02d}"
        _write_vasp_task(Path(directory) / name, atoms)
        tasks.append({"name": name, "kind": "eos", "lattice_parameter": float(lattice_parameter)})
    return material_name, symbol, crystal_structure, tasks


def build_followup_tasks(symbol, crystal_structure, equilibrium_lattice, directory):
    from ase.build import bulk, surface

    directory = Path(directory)
    tasks = []
    for index, strain in enumerate(STRAINS):
        normal_atoms = bulk(symbol, crystal_structure, a=equilibrium_lattice, cubic=True)
        normal_cell = np.diag([
            equilibrium_lattice * (1.0 + strain),
            equilibrium_lattice,
            equilibrium_lattice,
        ])
        normal_atoms.set_cell(normal_cell, scale_atoms=True)
        normal_name = f"normal_{index:02d}"
        _write_vasp_task(directory / normal_name, normal_atoms)
        tasks.append({"name": normal_name, "kind": "normal_stress", "strain": float(strain)})

        shear_atoms = bulk(symbol, crystal_structure, a=equilibrium_lattice, cubic=True)
        shear_cell = np.array([
            [equilibrium_lattice, strain * equilibrium_lattice, 0.0],
            [0.0, equilibrium_lattice, 0.0],
            [0.0, 0.0, equilibrium_lattice],
        ])
        shear_atoms.set_cell(shear_cell, scale_atoms=True)
        shear_name = f"shear_{index:02d}"
        _write_vasp_task(directory / shear_name, shear_atoms)
        tasks.append({"name": shear_name, "kind": "shear_stress", "strain": float(strain)})

    bulk_atoms = bulk(symbol, crystal_structure, a=equilibrium_lattice, cubic=True)
    for indices in SURFACE_ORIENTATIONS[crystal_structure]:
        slab = surface(
            bulk_atoms,
            indices,
            layers=VASP_SURFACE_LAYERS,
            vacuum=VASP_SURFACE_VACUUM,
            periodic=True,
        )
        slab.pbc = (True, True, True)
        name = "surface_" + "".join(str(index) for index in indices)
        _write_vasp_task(directory / name, slab, relax=True, kpoints=VASP_KPOINTS_SURFACE)
        tasks.append({
            "name": name,
            "kind": "surface",
            "miller_indices": tuple(indices),
            "orientation": "(" + "".join(str(index) for index in indices) + ")",
            "atom_count": len(slab),
            "area_A2": float(np.linalg.norm(np.cross(slab.cell.array[0], slab.cell.array[1]))),
        })
    return tasks


def _read_vasp_result(outcar_path):
    from ase.io import read

    atoms = read(str(outcar_path), index=-1, format="vasp-out")
    result = {"energy": float(atoms.get_potential_energy())}
    try:
        result["stress"] = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    except (RuntimeError, ValueError):
        result["stress"] = None
    try:
        result["forces"] = np.asarray(atoms.get_forces(), dtype=float)
    except (RuntimeError, ValueError):
        result["forces"] = None
    return result


def _fit_eos(material, lattice_structure, eos_tasks, eos_outputs):
    from ase.build import bulk

    _, symbol = _normalize_material(material)
    crystal_structure = _normalize_structure(lattice_structure)
    lattice_parameters = np.asarray([task["lattice_parameter"] for task in eos_tasks], dtype=float)
    energies = np.asarray([eos_outputs[task["name"]]["energy"] for task in eos_tasks], dtype=float)
    volumes = np.asarray([
        bulk(symbol, crystal_structure, a=float(parameter), cubic=True).get_volume()
        for parameter in lattice_parameters
    ], dtype=float)
    initial_index = int(np.argmin(energies))
    return curve_fit(
        birch_murnaghan,
        volumes,
        energies,
        p0=[energies[initial_index], volumes[initial_index], 0.4, 4.0],
        maxfev=100000,
    )[0]


def _remote_command(client, command, *, timeout=30, check=True):
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", errors="replace").strip()
    errors = stderr.read().decode("utf-8", errors="replace").strip()
    status = stdout.channel.recv_exit_status()
    if check and status != 0:
        detail = errors or output or f"remote command exited with status {status}"
        raise RuntimeError(detail[-6000:])
    return output, errors, status


def _connect(username, password):
    import paramiko

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(
            hostname=SSH_HOST,
            username=username,
            password=password,
            timeout=20,
            auth_timeout=30,
            banner_timeout=30,
            allow_agent=False,
            look_for_keys=False,
        )
    except Exception:
        client.close()
        raise
    return client


def _validate_remote_environment(client, symbol):
    quoted_symbol = shlex.quote(symbol)
    script = "\n".join((
        f"module use {shlex.quote(MODULEFILE_DIRECTORY)}",
        f"module load {' '.join(map(shlex.quote, VASP_PREREQUISITE_MODULES))}",
        f"module load {shlex.quote(VASP_MODULE)}",
        "vasp_executable=$(command -v vasp_std || true)",
        "printf 'VASP_EXECUTABLE=%s\\n' \"$vasp_executable\"",
        "printf 'VASP_PP_PATH=%s\\n' \"${VASP_PP_PATH:-}\"",
        "potcar_source=''",
        f"for candidate in \"${{VASP_PP_PATH:-}}\"/potpaw_PBE*/{quoted_symbol}/POTCAR \"${{VASP_PP_PATH:-}}\"/potpaw_PBE*/{quoted_symbol}*/POTCAR \"${{VASP_PP_PATH:-}}\"/{quoted_symbol}/POTCAR \"${{VASP_PP_PATH:-}}\"/{quoted_symbol}*/POTCAR; do if [ -f \"$candidate\" ]; then potcar_source=$candidate; break; fi; done",
        "printf 'POTCAR_SOURCE=%s\\n' \"$potcar_source\"",
        "test -n \"$vasp_executable\" && test -n \"$potcar_source\"",
    ))
    command = "bash -lc " + shlex.quote(script)
    output, errors, status = _remote_command(client, command, timeout=60, check=False)
    if status != 0:
        raise RuntimeError(
            "Cluster preflight failed for this element. Expected vasp_std and a PBE POTCAR "
            f"for {symbol} after loading {VASP_MODULE}.\n{output}\n{errors}".strip()
        )
    return output


def _stage_phase(sftp, client, remote_root, phase_name, tasks, symbol, local_phase, job_name, progress, poll_interval, job_timeout):
    remote_phase = f"{remote_root}/{phase_name}"
    _remote_command(client, f"mkdir -m 700 -p {shlex.quote(remote_phase)}")
    _remote_command(client, f"mkdir -m 700 -p {shlex.quote(remote_phase + '/tasks')}")
    run_script = _run_script_path()
    if not run_script.is_file():
        raise FileNotFoundError(f"Bundled VASP launcher not found: {run_script}")
    sftp.put(str(run_script), f"{remote_phase}/run.sh")
    sftp.putfo(_string_file(symbol + "\n"), f"{remote_phase}/.element")

    output_directories = {}
    for task in tasks:
        local_task = Path(local_phase) / task["name"]
        remote_task = f"{remote_phase}/tasks/{task['name']}"
        _remote_command(client, f"mkdir -m 700 {shlex.quote(remote_task)}")
        for filename in ("POSCAR", "KPOINTS", "INCAR.template"):
            sftp.put(str(local_task / filename), f"{remote_task}/{filename}")
        output_directories[task["name"]] = local_task / "OUTCAR"

    progress(f"Submitting {phase_name} VASP tasks to Slurm...")
    submit_command = (
        f"cd {shlex.quote(remote_phase)} && sbatch --parsable "
        f"--job-name={shlex.quote(job_name)} run.sh"
    )
    submit_output, _, _ = _remote_command(client, submit_command, timeout=60)
    match = re.search(r"(?:Submitted batch job\s+)?(\d+)", submit_output)
    if match is None:
        raise RuntimeError(f"Could not parse Slurm job ID from: {submit_output}")
    job_id = match.group(1)
    _wait_for_job(client, remote_phase, job_id, progress, poll_interval, job_timeout)

    for task in tasks:
        local_outcar = output_directories[task["name"]]
        remote_outcar = f"{remote_phase}/tasks/{task['name']}/OUTCAR"
        try:
            sftp.get(remote_outcar, str(local_outcar))
        except OSError as error:
            raise RuntimeError(f"Slurm job {job_id} completed without {task['name']}/OUTCAR: {error}") from error
    progress(f"Downloaded outputs from Slurm job {job_id}.")
    return {
        task["name"]: _read_vasp_result(output_directories[task["name"]])
        for task in tasks
    }


def _string_file(value):
    import io

    return io.BytesIO(value.encode("utf-8"))


def _wait_for_job(client, remote_phase, job_id, progress, poll_interval, job_timeout):
    failed_states = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
    deadline = time.monotonic() + job_timeout
    while time.monotonic() < deadline:
        queue_output, _, _ = _remote_command(
            client,
            f"squeue --noheader --jobs={shlex.quote(job_id)} --format=%T",
            check=False,
        )
        state = queue_output.splitlines()[0].strip().split("+")[0] if queue_output else ""
        if not state:
            account_output, _, _ = _remote_command(
                client,
                f"sacct --noheader --parsable2 --jobs={shlex.quote(job_id)} --format=State -X | head -n 1",
                check=False,
            )
            state = account_output.split("|", 1)[0].strip().split("+")[0]
        if state == "COMPLETED":
            return
        if state in failed_states:
            logs, _, _ = _remote_command(
                client,
                f"for file in {shlex.quote(remote_phase)}/slurm-{shlex.quote(job_id)}.* "
                f"{shlex.quote(remote_phase)}/tasks/*/vasp.log; do "
                "[ -f \"$file\" ] && { echo \"--- $file ---\"; tail -n 60 \"$file\"; }; done | tail -n 240",
                check=False,
            )
            raise RuntimeError(f"Slurm job {job_id} ended in state {state}.\n{logs}".strip())
        progress(f"Slurm job {job_id}: {state or 'waiting for accounting status'}...")
        time.sleep(poll_interval)

    _remote_command(client, f"scancel {shlex.quote(job_id)}", check=False)
    raise TimeoutError(f"Slurm job {job_id} exceeded the {job_timeout}-second wait limit and was cancelled.")


def _assemble_report(material_name, symbol, lattice_structure, eos_tasks, eos_outputs, followup_tasks, followup_outputs):
    from ase.build import bulk

    crystal_structure = _normalize_structure(lattice_structure)
    lattice_parameters = np.asarray([task["lattice_parameter"] for task in eos_tasks], dtype=float)
    energies = np.asarray([eos_outputs[task["name"]]["energy"] for task in eos_tasks], dtype=float)
    volumes = np.asarray([
        bulk(symbol, crystal_structure, a=float(parameter), cubic=True).get_volume()
        for parameter in lattice_parameters
    ], dtype=float)
    atom_count = len(bulk(symbol, crystal_structure, a=float(lattice_parameters[0]), cubic=True))
    initial_index = int(np.argmin(energies))
    fit_parameters, _ = curve_fit(
        birch_murnaghan,
        volumes,
        energies,
        p0=[energies[initial_index], volumes[initial_index], 0.4, 4.0],
        maxfev=100000,
    )
    energy0, volume0, bulk_modulus_eva3, bulk_modulus_prime = fit_parameters
    strains = STRAINS.copy()
    normal_by_strain = {task["strain"]: followup_outputs[task["name"]]["stress"] for task in followup_tasks if task["kind"] == "normal_stress"}
    shear_by_strain = {task["strain"]: followup_outputs[task["name"]]["stress"] for task in followup_tasks if task["kind"] == "shear_stress"}
    sigma_xx = np.asarray([normal_by_strain[float(strain)][0] for strain in strains])
    sigma_yy = np.asarray([normal_by_strain[float(strain)][1] for strain in strains])
    sigma_xy = np.asarray([shear_by_strain[float(strain)][5] for strain in strains])
    c11_fit = fit_linear_response(strains, sigma_xx)
    c12_fit = fit_linear_response(strains, sigma_yy)
    c44_fit = fit_linear_response(strains, sigma_xy)
    c11, c12, c44 = (fit[0] * EV_A3_TO_GPA for fit in (c11_fit, c12_fit, c44_fit))
    derived = derive_cubic_properties(c11, c12, c44)
    equilibrium_lattice = float(np.cbrt(volume0))
    surface_results = []
    for task in followup_tasks:
        if task["kind"] != "surface":
            continue
        output = followup_outputs[task["name"]]
        forces = output["forces"]
        converged = bool(
            forces is not None
            and len(forces)
            and np.max(np.linalg.norm(forces, axis=1)) <= VASP_SURFACE_FMAX
        )
        surface_results.append({
            "orientation": task["orientation"],
            "miller_indices": task["miller_indices"],
            "energy_eV_A2": float(
                (output["energy"] - task["atom_count"] * energy0 / atom_count)
                / (2.0 * task["area_A2"])
            ),
            "slab_energy_eV": output["energy"],
            "bulk_energy_per_atom_eV": float(energy0 / atom_count),
            "atom_count": task["atom_count"],
            "area_A2": task["area_A2"],
            "converged": converged,
        })

    return {
        "material": material_name,
        "symbol": symbol,
        "lattice_structure": lattice_structure,
        "method": "DFT",
        "model": None,
        "atoms_per_cell": atom_count,
        "a0": equilibrium_lattice,
        "V0": float(volume0),
        "E0_atom": float(energy0 / atom_count),
        "B_EOS": float(bulk_modulus_eva3 * EV_A3_TO_GPA),
        "B_prime": float(bulk_modulus_prime),
        "C11": float(c11),
        "C12": float(c12),
        "C44": float(c44),
        **derived,
        "R2_C11": c11_fit[2],
        "R2_C12": c12_fit[2],
        "R2_C44": c44_fit[2],
        "eos_data": {
            "lattice_parameter": lattice_parameters,
            "volume": volumes,
            "energy_per_atom": energies / atom_count,
        },
        "elastic_data": {
            "strain": strains,
            "sigma_xx_GPa": sigma_xx * EV_A3_TO_GPA,
            "sigma_yy_GPa": sigma_yy * EV_A3_TO_GPA,
            "sigma_xy_GPa": sigma_xy * EV_A3_TO_GPA,
        },
        "surface_energy_data": {
            "material": material_name,
            "symbol": symbol,
            "lattice_structure": lattice_structure,
            "method": "DFT",
            "surfaces": surface_results,
        },
    }


def calculate_remote_dft_report(
    material,
    lattice_structure,
    username,
    password,
    *,
    progress=None,
    poll_interval=15,
    job_timeout=4 * 60 * 60,
):
    """Run EOS, elasticity, and surface calculations using remote VASP jobs."""
    from ase.build import bulk

    progress = progress or (lambda message: None)
    material_name, symbol = _normalize_material(material)
    crystal_structure = _normalize_structure(lattice_structure)
    if not username.strip() or not password:
        raise ValueError("Cluster username and password are required for DFT.")

    client = None
    remote_root = f"{REMOTE_SCRATCH_DIRECTORY.rstrip('/')}/materials-property-{uuid.uuid4().hex}"
    try:
        progress(f"Connecting to {SSH_HOST}...")
        client = _connect(username.strip(), password)
        sftp = client.open_sftp()
        try:
            preflight = _validate_remote_environment(client, symbol)
            progress(f"VASP {VASP_MODULE} and {symbol} POTCAR found on the cluster.")
            _remote_command(client, f"mkdir -m 700 {shlex.quote(remote_root)}")
            with tempfile.TemporaryDirectory(prefix="materials-property-") as temporary_directory:
                local_root = Path(temporary_directory)
                eos_directory = local_root / "eos"
                _, _, _, eos_tasks = build_eos_tasks(material_name, lattice_structure, eos_directory)
                eos_outputs = _stage_phase(
                    sftp,
                    client,
                    remote_root,
                    "eos",
                    eos_tasks,
                    symbol,
                    eos_directory,
                    f"dft-{symbol.lower()}-eos",
                    progress,
                    poll_interval,
                    job_timeout,
                )
                fit_parameters = _fit_eos(material_name, lattice_structure, eos_tasks, eos_outputs)
                equilibrium_lattice = float(np.cbrt(fit_parameters[1]))
                progress(f"EOS fit complete; equilibrium lattice parameter {equilibrium_lattice:.6f} Å.")

                followup_directory = local_root / "followup"
                followup_tasks = build_followup_tasks(
                    symbol,
                    crystal_structure,
                    equilibrium_lattice,
                    followup_directory,
                )
                followup_outputs = _stage_phase(
                    sftp,
                    client,
                    remote_root,
                    "followup",
                    followup_tasks,
                    symbol,
                    followup_directory,
                    f"dft-{symbol.lower()}-elastic",
                    progress,
                    poll_interval,
                    job_timeout,
                )
                return _assemble_report(
                    material_name,
                    symbol,
                    lattice_structure,
                    eos_tasks,
                    eos_outputs,
                    followup_tasks,
                    followup_outputs,
                )
        finally:
            sftp.close()
    finally:
        if client is not None:
            try:
                _remote_command(client, f"rm -rf -- {shlex.quote(remote_root)}", timeout=60, check=False)
            finally:
                client.close()
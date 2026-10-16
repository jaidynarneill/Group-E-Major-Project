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
        VASP_MODULE,
        clean_shell_startup_warnings,
        module_shell_command,
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
        VASP_MODULE,
        clean_shell_startup_warnings,
        module_shell_command,
    )


VASP_WORKFLOW_VERSION = 7
VASP_MPI_TASKS = 16
VASP_KPOINTS_DENSITY = 1464
VASP_ENCUT_FACTOR = 1.3
VASP_BULK_RELAX_MAX_STEPS = 99
VASP_INITIAL_MAGMOM = {"Co": 0.6}
VASP_SURFACE_LAYERS = 8
VASP_SURFACE_VACUUM = 10.0
VASP_SURFACE_FMAX = 0.05
VASP_SURFACE_MAX_STEPS = 200
VASP_CACHE_BACKEND = (
    "VASP",
    VASP_MODULE,
    VASP_WORKFLOW_VERSION,
    VASP_MPI_TASKS,
    VASP_KPOINTS_DENSITY,
    VASP_ENCUT_FACTOR,
    VASP_BULK_RELAX_MAX_STEPS,
    tuple(sorted(VASP_INITIAL_MAGMOM.items())),
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


def _remote_material_directory(material_name):
    folder_name = re.sub(r"[^A-Za-z0-9]+", "-", material_name).strip("-")
    if not folder_name:
        raise ValueError("Material name cannot produce a remote folder name.")
    return f"{REMOTE_SCRATCH_DIRECTORY.rstrip('/')}/{folder_name}-Dft"


def _remote_run_directory(material_name, run_id=None):
    run_id = run_id or uuid.uuid4().hex
    return f"{_remote_material_directory(material_name)}/run-{run_id}"


def _kpoint_mesh(atoms):
    reciprocal_lengths = np.linalg.norm(atoms.cell.reciprocal(), axis=1)
    geometric_mean = float(np.cbrt(np.prod(reciprocal_lengths)))
    density_per_atom = VASP_KPOINTS_DENSITY / len(atoms)
    scaled_mesh = np.cbrt(density_per_atom) * reciprocal_lengths / geometric_mean
    return tuple(np.maximum(1, np.rint(scaled_mesh).astype(int)))


def _write_vasp_task(directory, atoms, *, relax=False, relax_cell=False):
    from ase.io import write

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / "POSCAR", atoms, format="vasp", direct=True, vasp5=True, sort=True)
    mesh = _kpoint_mesh(atoms)
    (directory / "KPOINTS").write_text(
        "Automatic mesh (grid density = 1464 / number of atoms)\n0\nGamma\n"
        + " ".join(map(str, mesh))
        + "\n",
        encoding="ascii",
    )
    incar = [
        "PREC = Accurate",
        "ENCUT = @ENCUT@",
        "EDIFF = 2E-4" if relax else "EDIFF = 1E-8",
        "ISMEAR = 0",
        "SIGMA = 0.05",
        "LREAL = Auto" if relax_cell else "LREAL = .FALSE.",
        "LASPH = .TRUE.",
        "ISYM = 0",
        f"ISIF = {3 if relax_cell else 2}",
        f"IBRION = {2 if relax else -1}",
        f"NSW = {VASP_BULK_RELAX_MAX_STEPS if relax_cell else VASP_SURFACE_MAX_STEPS if relax else 0}",
        "LWAVE = .FALSE.",
        "LCHARG = .FALSE.",
    ]
    symbol = atoms.get_chemical_symbols()[0]
    if symbol in VASP_INITIAL_MAGMOM:
        incar.extend((
            "ISPIN = 2",
            f"MAGMOM = {len(atoms)}*{VASP_INITIAL_MAGMOM[symbol]}",
            "LMAXMIX = 4",
            "LORBIT = 11",
            "ALGO = Fast",
        ))
    if relax:
        incar.append(f"EDIFFG = -{VASP_SURFACE_FMAX}")
        if not relax_cell:
            incar.append("AMIN = 0.01")
            incar.append("KPAR = 4")
    (directory / "INCAR.template").write_text("\n".join(incar) + "\n", encoding="ascii")


def build_bulk_relaxation_task(material, lattice_structure, directory):
    from ase.build import bulk

    _, symbol = _normalize_material(material)
    crystal_structure = _normalize_structure(lattice_structure)
    atoms = bulk(
        symbol,
        crystal_structure,
        a=REFERENCE_LATTICE_PARAMETERS[symbol],
        cubic=True,
    )
    name = "bulk_relax"
    _write_vasp_task(Path(directory) / name, atoms, relax=True, relax_cell=True)
    return {"name": name, "kind": "bulk_relax", "atom_count": len(atoms)}


def build_eos_tasks(material, lattice_structure, directory, *, reference_lattice=None):
    from ase.build import bulk

    material_name, symbol = _normalize_material(material)
    crystal_structure = _normalize_structure(lattice_structure)
    reference = reference_lattice or REFERENCE_LATTICE_PARAMETERS[symbol]
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
        _write_vasp_task(directory / name, slab, relax=True)
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
    result = {
        "energy": float(atoms.get_potential_energy()),
        "volume_A3": float(atoms.get_volume()),
    }
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
    script = "\n".join((
        f"module use {shlex.quote(MODULEFILE_DIRECTORY)}",
        f"module load {shlex.quote(VASP_MODULE)}",
        "vasp_executable=$(command -v vasp_std || true)",
        "potgen_executable=$(command -v potgen || true)",
        "printf 'VASP_EXECUTABLE=%s\\n' \"$vasp_executable\"",
        "printf 'POTGEN_EXECUTABLE=%s\\n' \"$potgen_executable\"",
        "test -n \"$vasp_executable\" && test -n \"$potgen_executable\"",
    ))
    command = module_shell_command(script)
    output, errors, status = _remote_command(client, command, timeout=60, check=False)
    errors = clean_shell_startup_warnings(errors)
    if status != 0:
        raise RuntimeError(
            "Cluster preflight failed. Expected vasp_std and potgen after loading "
            f"{VASP_MODULE} for {symbol}.\n{output}\n{errors}".strip()
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


def _job_log_tail_command(remote_phase, job_id):
    phase_path = shlex.quote(remote_phase)
    quoted_job_id = shlex.quote(job_id)
    return (
        f"for file in {phase_path}/tasks/*/vasp.log {phase_path}/tasks/*/OUTCAR "
        f"{phase_path}/slurm-*-{quoted_job_id}.out {phase_path}/slurm-*-{quoted_job_id}.err; do "
        "if [ -f \"$file\" ]; then printf '\\n--- %s ---\\n' \"$file\"; "
        "tail -n 12 \"$file\"; fi; done | tail -n 80"
    )


def _job_accounting_command(job_id):
    return (
        "sacct --noheader --parsable2 "
        f"--jobs={shlex.quote(job_id)} "
        "--format=State,ExitCode,MaxRSS,ReqMem,Elapsed -X | head -n 1"
    )


def _read_job_log_tail(client, remote_phase, job_id):
    output, _, _ = _remote_command(
        client,
        _job_log_tail_command(remote_phase, job_id),
        check=False,
    )
    return output


def _read_job_accounting(client, job_id):
    output, _, _ = _remote_command(
        client,
        _job_accounting_command(job_id),
        check=False,
    )
    return output or "unavailable"


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
            progress(f"Slurm job {job_id}: COMPLETED")
            return
        if state in failed_states:
            logs = _read_job_log_tail(client, remote_phase, job_id)
            accounting = _read_job_accounting(client, job_id)
            message = (
                f"Slurm job {job_id} ended in state {state}.\n"
                f"Slurm accounting (State|ExitCode|MaxRSS|ReqMem|Elapsed): {accounting}\n{logs}"
            ).strip()
            raise RuntimeError(message)
        message = f"Slurm job {job_id}: {state or 'waiting for accounting status'}..."
        progress(message)
        time.sleep(poll_interval)

    _remote_command(client, f"scancel {shlex.quote(job_id)}", check=False)
    logs = _read_job_log_tail(client, remote_phase, job_id)
    raise TimeoutError(
        f"Slurm job {job_id} exceeded the {job_timeout}-second wait limit and was cancelled.\n{logs}".strip()
    )


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
    progress = progress or (lambda message: None)
    material_name, symbol = _normalize_material(material)
    crystal_structure = _normalize_structure(lattice_structure)
    if not username.strip() or not password:
        raise ValueError("Cluster username and password are required for DFT.")

    client = None
    remote_material_root = _remote_material_directory(material_name)
    remote_root = _remote_run_directory(material_name)
    try:
        progress(f"Connecting to {SSH_HOST}...")
        client = _connect(username.strip(), password)
        sftp = client.open_sftp()
        try:
            preflight = _validate_remote_environment(client, symbol)
            progress(f"VASP {VASP_MODULE} and potgen found on the cluster.")
            _remote_command(
                client,
                f"rm -rf -- {shlex.quote(REMOTE_SCRATCH_DIRECTORY.rstrip('/') + '/slurm')}",
            )
            _remote_command(client, f"mkdir -m 700 -p {shlex.quote(remote_material_root)}")
            _remote_command(client, f"mkdir -m 700 {shlex.quote(remote_root)}")
            with tempfile.TemporaryDirectory(prefix="materials-property-") as temporary_directory:
                local_root = Path(temporary_directory)
                relaxation_directory = local_root / "bulk_relax"
                relaxation_task = build_bulk_relaxation_task(
                    material_name,
                    lattice_structure,
                    relaxation_directory,
                )
                relaxation_outputs = _stage_phase(
                    sftp,
                    client,
                    remote_root,
                    "bulk-relax",
                    [relaxation_task],
                    symbol,
                    relaxation_directory,
                    f"dft-{symbol.lower()}-relax",
                    progress,
                    poll_interval,
                    job_timeout,
                )
                relaxed_volume = relaxation_outputs[relaxation_task["name"]]["volume_A3"]
                if not np.isfinite(relaxed_volume) or relaxed_volume <= 0.0:
                    raise RuntimeError("Bulk relaxation returned an invalid cell volume.")
                relaxed_lattice = float(np.cbrt(relaxed_volume))
                progress(f"Bulk relaxation complete; starting EOS around {relaxed_lattice:.6f} Å.")

                eos_directory = local_root / "eos"
                _, _, _, eos_tasks = build_eos_tasks(
                    material_name,
                    lattice_structure,
                    eos_directory,
                    reference_lattice=relaxed_lattice,
                )
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
            client.close()
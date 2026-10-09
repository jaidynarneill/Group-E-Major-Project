import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


EV_A3_TO_GPA = 160.21766208
STRAINS = np.array([-0.005, -0.0025, -0.001, 0.001, 0.0025, 0.005])
DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
REFERENCE_LATTICE_PARAMETERS = {"Al": 4.05, "Cu": 3.615, "Si": 5.43}
CRYSTAL_STRUCTURES = {
    "fcc": "fcc",
    "bcc": "bcc",
    "dia": "diamond",
    "diamond": "diamond",
    "diamond cubic": "diamond",
    "sc": "sc",
    "simple cubic": "sc",
}
MEAM_LATTICE_NAMES = {
    "fcc": "FCC",
    "bcc": "BCC",
    "dia": "Diamond cubic",
    "sc": "Simple cubic",
}
ELEMENT_SYMBOLS = {
    3: "Li", 6: "C", 11: "Na", 12: "Mg", 13: "Al", 14: "Si",
    19: "K", 23: "V", 24: "Cr", 26: "Fe", 28: "Ni", 29: "Cu",
    41: "Nb", 42: "Mo", 45: "Rh", 46: "Pd", 47: "Ag", 50: "Sn",
    73: "Ta", 74: "W", 77: "Ir", 78: "Pt", 79: "Au", 82: "Pb", 83: "Bi",
}
ELEMENT_NAMES = {
    "Li": "Lithium", "C": "Carbon", "Na": "Sodium", "Mg": "Magnesium",
    "Al": "Aluminum", "Si": "Silicon", "K": "Potassium", "V": "Vanadium",
    "Cr": "Chromium", "Fe": "Iron", "Ni": "Nickel", "Cu": "Copper",
    "Nb": "Niobium", "Mo": "Molybdenum", "Rh": "Rhodium", "Pd": "Palladium",
    "Ag": "Silver", "Sn": "Tin", "Ta": "Tantalum", "W": "Tungsten",
    "Ir": "Iridium", "Pt": "Platinum", "Au": "Gold", "Pb": "Lead", "Bi": "Bismuth",
}
MACE_MODEL_OPTIONS = {
    "MACE-MP Small": "small",
    "MACE-MP Medium": "medium",
    "MACE-MP Large": "large",
    "MACE-MPA": "medium-mpa-0",
}


def mace_model_for_method(method, default="small"):
    key = re.sub(r"[\s_]+", "-", str(method).strip().upper())
    for label, model_name in MACE_MODEL_OPTIONS.items():
        if key == re.sub(r"[\s_]+", "-", label.upper()):
            return model_name
    return default


def load_meam_library_catalog():
    """Read supported elemental cubic records from the bundled LAMMPS library."""
    library_path = DATA_DIRECTORY / "potentials" / "library.meam"
    if not library_path.is_file():
        return {}

    candidates = {}
    lines = library_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines[:-1]):
        match = re.match(r"\s*'([^']+)'\s+'([^']+)'\s+[\d.]+\s+(\d+)\s+[\d.]+", line)
        if match is None:
            continue

        library_element, lattice_code, atomic_number_text = match.groups()
        lattice_code = lattice_code.lower()
        atomic_number = int(atomic_number_text)
        symbol = ELEMENT_SYMBOLS.get(atomic_number)
        if symbol is None or lattice_code not in MEAM_LATTICE_NAMES:
            continue

        parameter_values = lines[index + 1].split()
        try:
            lattice_parameter = float(parameter_values[5])
        except (IndexError, ValueError):
            continue
        if lattice_parameter <= 0.0:
            continue

        candidates.setdefault(symbol, []).append({
            "symbol": symbol,
            "name": ELEMENT_NAMES[symbol],
            "library_element": library_element,
            "lattice": CRYSTAL_STRUCTURES[lattice_code],
            "lattice_name": MEAM_LATTICE_NAMES[lattice_code],
            "lattice_parameter": lattice_parameter,
            "atomic_number": atomic_number,
        })

    catalog = {}
    for symbol, records in sorted(candidates.items(), key=lambda item: item[1][0]["atomic_number"]):
        canonical_records = [record for record in records if record["library_element"] == symbol]
        catalog[symbol] = canonical_records[0] if canonical_records else records[0]
    return catalog


MEAM_CATALOG = load_meam_library_catalog()
REFERENCE_LATTICE_PARAMETERS.update({
    symbol: record["lattice_parameter"] for symbol, record in MEAM_CATALOG.items()
})


def birch_murnaghan(volume, energy0, volume0, bulk_modulus, bulk_modulus_prime):
    eta = (volume0 / volume) ** (2.0 / 3.0)
    return energy0 + (9.0 * volume0 * bulk_modulus / 16.0) * (
        (eta - 1.0) ** 3 * bulk_modulus_prime
        + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def fit_linear_response(strain, response):
    """Return slope, intercept, and R-squared for a linear stress-strain fit."""
    strain = np.asarray(strain, dtype=float)
    response = np.asarray(response, dtype=float)
    if strain.ndim != 1 or response.ndim != 1 or strain.size != response.size:
        raise ValueError("Strain and response must be one-dimensional arrays of equal length.")
    if strain.size < 2 or not np.isfinite(strain).all() or not np.isfinite(response).all():
        raise ValueError("At least two finite strain-response data points are required.")

    slope, intercept = np.polyfit(strain, response, 1)
    fitted = slope * strain + intercept
    residual_sum = np.sum((response - fitted) ** 2)
    total_sum = np.sum((response - np.mean(response)) ** 2)
    if total_sum == 0.0:
        r_squared = 1.0 if residual_sum == 0.0 else 0.0
    else:
        r_squared = 1.0 - residual_sum / total_sum
    return float(slope), float(intercept), float(r_squared)


def derive_cubic_properties(c11, c12, c44):
    """Calculate bulk, Voigt-Reuss-Hill, Young's, Poisson, and Zener values."""
    bulk_modulus = (c11 + 2.0 * c12) / 3.0
    voigt_shear = (c11 - c12 + 3.0 * c44) / 5.0
    reuss_denominator = 4.0 * c44 + 3.0 * (c11 - c12)
    if reuss_denominator == 0.0 or c11 == c12:
        raise ValueError("Elastic constants do not define finite cubic aggregate properties.")

    reuss_shear = 5.0 * (c11 - c12) * c44 / reuss_denominator
    hill_shear = (voigt_shear + reuss_shear) / 2.0
    young_denominator = 3.0 * bulk_modulus + hill_shear
    if young_denominator == 0.0:
        raise ValueError("Elastic constants do not define a finite Young's modulus.")
    young_modulus = 9.0 * bulk_modulus * hill_shear / young_denominator
    poisson_ratio = (3.0 * bulk_modulus - 2.0 * hill_shear) / (2.0 * young_denominator)

    return {
        "B_Cij": bulk_modulus,
        "G_V": voigt_shear,
        "G_R": reuss_shear,
        "G_H": hill_shear,
        "E": young_modulus,
        "nu": poisson_ratio,
        "A": 2.0 * c44 / (c11 - c12),
    }


def _normalize_material(material):
    key = str(material).strip().lower()
    materials = {
        "al": ("Aluminum", "Al"),
        "aluminium": ("Aluminum", "Al"),
        "aluminum": ("Aluminum", "Al"),
        "cu": ("Copper", "Cu"),
        "copper": ("Copper", "Cu"),
        "si": ("Silicon", "Si"),
        "silicon": ("Silicon", "Si"),
    }
    for symbol, record in MEAM_CATALOG.items():
        materials[symbol.lower()] = (record["name"], symbol)
        materials[record["name"].lower()] = (record["name"], symbol)
    try:
        return materials[key]
    except KeyError as error:
        raise ValueError(f"Unsupported material {material!r}; choose aluminum, copper, or silicon.") from error


def _normalize_structure(lattice_structure):
    key = " ".join(str(lattice_structure).strip().lower().replace("-", " ").split())
    try:
        return CRYSTAL_STRUCTURES[key]
    except KeyError as error:
        supported = ", ".join(sorted(CRYSTAL_STRUCTURES))
        raise ValueError(
            f"Unsupported lattice structure {lattice_structure!r}. "
            f"Cubic elasticity fits currently support: {supported}."
        ) from error


def _normalize_method(method):
    key = re.sub(r"[\s_]+", "-", str(method).strip().upper())
    mace_methods = {
        "MACE", "MACE-MP", "MACE-MP-SMALL", "MACE-MP-MEDIUM",
        "MACE-MP-LARGE", "MACE-MPA",
    }
    if key in mace_methods:
        return "MACE-MP"
    aliases = {"MEAM": "MEAM", "DFT": "DFT"}
    try:
        return aliases[key]
    except KeyError as error:
        raise ValueError("Method must be MEAM, MACE-MP, or DFT.") from error


def _find_meam_file(explicit_path, environment_name, search_pattern):
    configured_path = explicit_path or os.getenv(environment_name)
    if configured_path:
        path = Path(configured_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configured {environment_name} file does not exist: {path}")
        return path

    potential_directory = DATA_DIRECTORY / "potentials"
    matches = sorted(potential_directory.glob(search_pattern)) if potential_directory.is_dir() else []
    return matches[0] if len(matches) == 1 else None


def _create_calculator(
    method,
    symbol,
    model,
    meam_library,
    meam_parameter_file,
    meam_pair_coeff,
    meam_library_element,
    lammps_command,
):
    if method == "MACE-MP":
        try:
            from mace.calculators import mace_mp
        except ImportError as error:
            raise RuntimeError(
                "MACE-MP requires ASE and mace-torch. Install them with "
                "'python -m pip install ase mace-torch'."
            ) from error
        return mace_mp(model=model, default_dtype="float64")

    if method == "MEAM":
        library = _find_meam_file(meam_library, "MEAM_LIBRARY", "library*.meam")
        parameter = _find_meam_file(meam_parameter_file, "MEAM_PARAMETER_FILE", f"*{symbol}*.meam")
        custom_pair_coeff = meam_pair_coeff or os.getenv("MEAM_PAIR_COEFF")
        if library is None:
            raise FileNotFoundError(
                "MEAM library file not found. Set MEAM_LIBRARY or place library.meam "
                "under src/data/potentials/."
            )
        command = lammps_command or os.getenv("LAMMPS_COMMAND", "lmp")
        if not shutil.which(command) and not Path(command).is_file():
            raise RuntimeError(
                f"LAMMPS executable {command!r} was not found. Install LAMMPS and set "
                "LAMMPS_COMMAND to its executable path."
            )

        try:
            import ase.calculators.lammpsrun as lammpsrun
        except ImportError as error:
            raise RuntimeError("MEAM calculations require ASE; install it with 'python -m pip install ase'.") from error

        files = [str(library)] + ([str(parameter)] if parameter else [])
        library_element = meam_library_element or MEAM_CATALOG.get(
            symbol, {"library_element": symbol}
        )["library_element"]
        if custom_pair_coeff:
            pair_coeff = custom_pair_coeff.format(
                library=library.name,
                parameter=parameter.name if parameter else "NULL",
                element=symbol,
            )
        elif parameter:
            pair_coeff = f"* * {library.name} {library_element} {parameter.name} {symbol}"
        else:
            pair_coeff = f"* * {library.name} {library_element} NULL {library_element}"
        if os.name == "nt":
            from ase.calculators.calculator import Calculator, all_changes
            from ase.calculators.lammps import Prism, convert
            from ase.io.lammpsdata import write_lammps_data
            from ase.io.lammpsrun import read_lammps_dump

            class WindowsLAMMPSCalculator(Calculator):
                implemented_properties = ["energy", "free_energy", "forces", "stress"]

                def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                    super().calculate(atoms, properties, system_changes)
                    prism = Prism(atoms.cell.array, pbc=atoms.get_pbc())
                    with tempfile.TemporaryDirectory(prefix="materials-meam-") as directory:
                        working_directory = Path(directory)
                        for potential_file in files:
                            shutil.copy2(potential_file, working_directory / Path(potential_file).name)

                        data_name = "atoms.data"
                        dump_name = "forces.dump"
                        with (working_directory / data_name).open("w", encoding="ascii") as data_file:
                            write_lammps_data(
                                data_file,
                                atoms,
                                specorder=[symbol],
                                prismobj=prism,
                                units="metal",
                                atom_style="atomic",
                            )

                        input_script = "\n".join((
                            "clear",
                            "units metal",
                            "atom_style atomic",
                            "boundary p p p",
                            "box tilt large",
                            f"read_data {data_name}",
                            "pair_style meam",
                            f"pair_coeff {pair_coeff}",
                            f"mass 1 {atoms.get_masses()[0]:.12g}",
                            "thermo_style custom step pe pxx pyy pzz pyz pxz pxy",
                            "thermo 1",
                            f"dump results all custom 1 {dump_name} id type x y z fx fy fz",
                            "dump_modify results sort id",
                            "run 0",
                            "log none",
                            "",
                        ))

                        if Path(command).is_file():
                            command_parts = [str(Path(command).resolve())]
                        else:
                            command_parts = shlex.split(command, posix=os.name == "posix")
                            executable = shutil.which(command_parts[0])
                            if executable:
                                command_parts[0] = executable
                        completed = subprocess.run(
                            [*command_parts, "-echo", "none", "-log", "none"],
                            input=input_script,
                            text=True,
                            capture_output=True,
                            cwd=working_directory,
                            timeout=300,
                            check=False,
                        )
                        if completed.returncode != 0:
                            diagnostic = (completed.stdout + "\n" + completed.stderr).strip()
                            raise RuntimeError(
                                f"LAMMPS exited with code {completed.returncode}: {diagnostic[-4000:]}"
                            )

                        thermo_values = None
                        for line in completed.stdout.splitlines():
                            columns = line.split()
                            if len(columns) == 8:
                                try:
                                    values = [float(value) for value in columns]
                                except ValueError:
                                    continue
                                thermo_values = values
                        if thermo_values is None:
                            raise RuntimeError(
                                "LAMMPS completed without a parseable thermo row. "
                                + completed.stdout[-2000:]
                            )

                        dump_atoms = read_lammps_dump(
                            str(working_directory / dump_name),
                            order=True,
                            index=-1,
                            prismobj=prism,
                            specorder=[symbol],
                        )
                        pressure = np.array([
                            [-thermo_values[2], -thermo_values[7], -thermo_values[6]],
                            [-thermo_values[7], -thermo_values[3], -thermo_values[5]],
                            [-thermo_values[6], -thermo_values[5], -thermo_values[4]],
                        ])
                        stress = prism.tensor2_to_ase(pressure)
                        stress = stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
                        self.results = {
                            "energy": convert(thermo_values[1], "energy", "metal", "ASE"),
                            "free_energy": convert(thermo_values[1], "energy", "metal", "ASE"),
                            "forces": convert(dump_atoms.get_forces(), "force", "metal", "ASE"),
                            "stress": convert(stress, "pressure", "metal", "ASE"),
                        }

            return WindowsLAMMPSCalculator()

        return lammpsrun.LAMMPS(
            command=command,
            files=files,
            specorder=[symbol],
            units="metal",
            atom_style="atomic",
            pair_style="meam",
            pair_coeff=[pair_coeff],
        )

    if method == "DFT":
        raise RuntimeError("DFT calculations use the remote VASP Slurm workflow.")

    raise ValueError(f"Unsupported method: {method}")


def compute_elastic_properties(
    material,
    lattice_structure,
    method,
    *,
    model="small",
    lattice_parameters=None,
    strains=None,
    meam_library=None,
    meam_parameter_file=None,
    meam_pair_coeff=None,
    meam_library_element=None,
    lammps_command=None,
    calculator=None,
):
    """Run cubic EOS and small-strain stress calculations using an ASE calculator.

    MEAM uses LAMMPS and user-supplied potential files. DFT is executed remotely
    through the VASP workflow and is not an ASE calculator in this function.
    """
    material_name, symbol = _normalize_material(material)
    crystal_structure = _normalize_structure(lattice_structure)
    method_name = _normalize_method(method)
    if method_name == "DFT":
        raise RuntimeError("DFT calculations use calculate_remote_dft_report and remote VASP.")
    if method_name == "MACE-MP":
        model = mace_model_for_method(method, model)

    try:
        from ase.build import bulk
    except ImportError as error:
        raise RuntimeError("Elastic-property calculations require ASE; install it with 'python -m pip install ase'.") from error

    if calculator is None:
        calculator = _create_calculator(
            method_name,
            symbol,
            model,
            meam_library,
            meam_parameter_file,
            meam_pair_coeff,
            meam_library_element,
            lammps_command,
        )

    if lattice_parameters is None:
        reference = REFERENCE_LATTICE_PARAMETERS[symbol]
        lattice_parameters = np.linspace(reference * 0.97, reference * 1.03, 13)
    else:
        lattice_parameters = np.asarray(lattice_parameters, dtype=float)
    if lattice_parameters.ndim != 1 or lattice_parameters.size < 4:
        raise ValueError("At least four lattice parameters are required for the EOS fit.")
    if not np.isfinite(lattice_parameters).all() or np.any(lattice_parameters <= 0.0):
        raise ValueError("Lattice parameters must be finite positive values.")

    strains = STRAINS.copy() if strains is None else np.asarray(strains, dtype=float)
    if strains.ndim != 1 or strains.size < 2 or not np.isfinite(strains).all():
        raise ValueError("At least two finite strain values are required.")

    energies = []
    volumes = []
    atom_count = None
    for lattice_parameter in lattice_parameters:
        atoms = bulk(symbol, crystal_structure, a=float(lattice_parameter), cubic=True)
        atoms.calc = calculator
        energies.append(atoms.get_potential_energy())
        volumes.append(atoms.get_volume())
        atom_count = len(atoms)

    energies = np.asarray(energies, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    initial_index = int(np.argmin(energies))
    fit_parameters, _ = curve_fit(
        birch_murnaghan,
        volumes,
        energies,
        p0=[energies[initial_index], volumes[initial_index], 0.4, 4.0],
        maxfev=100000,
    )
    energy0, volume0, bulk_modulus_eva3, bulk_modulus_prime = fit_parameters
    equilibrium_lattice = float(np.cbrt(volume0))

    def get_stress(cell):
        atoms = bulk(symbol, crystal_structure, a=equilibrium_lattice, cubic=True)
        atoms.set_cell(cell, scale_atoms=True)
        atoms.calc = calculator
        return atoms.get_stress(voigt=True)

    sigma_xx = []
    sigma_yy = []
    sigma_xy = []
    for strain in strains:
        normal_cell = np.diag([
            equilibrium_lattice * (1.0 + strain),
            equilibrium_lattice,
            equilibrium_lattice,
        ])
        normal_stress = get_stress(normal_cell)
        sigma_xx.append(normal_stress[0])
        sigma_yy.append(normal_stress[1])

        shear_cell = np.array([
            [equilibrium_lattice, strain * equilibrium_lattice, 0.0],
            [0.0, equilibrium_lattice, 0.0],
            [0.0, 0.0, equilibrium_lattice],
        ])
        sigma_xy.append(get_stress(shear_cell)[5])

    sigma_xx = np.asarray(sigma_xx)
    sigma_yy = np.asarray(sigma_yy)
    sigma_xy = np.asarray(sigma_xy)
    c11_fit = fit_linear_response(strains, sigma_xx)
    c12_fit = fit_linear_response(strains, sigma_yy)
    c44_fit = fit_linear_response(strains, sigma_xy)
    c11, c12, c44 = (fit[0] * EV_A3_TO_GPA for fit in (c11_fit, c12_fit, c44_fit))
    derived = derive_cubic_properties(c11, c12, c44)

    return {
        "material": material_name,
        "symbol": symbol,
        "lattice_structure": lattice_structure,
        "method": method_name,
        "model": model if method_name == "MACE-MP" else None,
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
    }


def create_calculator(
    material,
    method,
    *,
    model="small",
    meam_library=None,
    meam_parameter_file=None,
    meam_pair_coeff=None,
    meam_library_element=None,
    lammps_command=None,
):
    """Create the requested ASE calculator for reuse across related calculations."""
    _, symbol = _normalize_material(material)
    method_name = _normalize_method(method)
    if method_name == "MACE-MP":
        model = mace_model_for_method(method, model)
    return _create_calculator(
        method_name,
        symbol,
        model,
        meam_library,
        meam_parameter_file,
        meam_pair_coeff,
        meam_library_element,
        lammps_command,
    )


def calculate_elastic_properties(material, method, lattice_structure=None, **options):
    """Compatibility wrapper defaulting to FCC metals and diamond-cubic silicon."""
    if lattice_structure is None:
        material_name, _ = _normalize_material(material)
        lattice_structure = "Diamond cubic" if material_name == "Silicon" else "FCC"
    return compute_elastic_properties(material, lattice_structure, method, **options)


def compare_elastic_properties(material1, material2, lattice_structure1, lattice_structure2, method, **options):
    """Compare two materials using the same calculation method."""
    return {
        "material1": compute_elastic_properties(material1, lattice_structure1, method, **options),
        "material2": compute_elastic_properties(material2, lattice_structure2, method, **options),
    }


def compare_configurations(configurations, **options):
    """Calculate each (material, lattice_structure, method) configuration in order."""
    return [
        compute_elastic_properties(material, lattice_structure, method, **options)
        for material, lattice_structure, method in configurations
    ]
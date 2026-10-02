import numpy as np

try:
    from analysis.elastic_properties import (
        DATA_DIRECTORY,
        REFERENCE_LATTICE_PARAMETERS,
        _normalize_material,
        _normalize_structure,
        create_calculator,
    )
except ModuleNotFoundError as error:
    if error.name != "analysis":
        raise
    from src.analysis.elastic_properties import (
        DATA_DIRECTORY,
        REFERENCE_LATTICE_PARAMETERS,
        _normalize_material,
        _normalize_structure,
        create_calculator,
    )


SURFACE_ORIENTATIONS = {
    "fcc": ((1, 0, 0), (1, 1, 0), (1, 1, 1)),
    "bcc": ((1, 0, 0), (1, 1, 0), (1, 1, 1)),
    "diamond": ((1, 0, 0), (1, 1, 0), (1, 1, 1)),
    "sc": ((1, 0, 0), (1, 1, 0), (1, 1, 1)),
}


def surface_energy_from_slab(slab_energy, bulk_energy_per_atom, atom_count, surface_area):
    """Return gamma in eV/A^2 for a slab with two equivalent exposed surfaces."""
    values = np.asarray([slab_energy, bulk_energy_per_atom, surface_area], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Surface and bulk energies and surface area must be finite.")
    if atom_count <= 0 or surface_area <= 0.0:
        raise ValueError("Atom count and surface area must be positive.")
    return float((slab_energy - atom_count * bulk_energy_per_atom) / (2.0 * surface_area))


def compute_surface_energies(
    material,
    lattice_structure,
    method,
    *,
    calculator=None,
    bulk_energy_per_atom=None,
    lattice_parameter=None,
    orientations=None,
    layers=8,
    vacuum=10.0,
    fmax=0.05,
    max_steps=200,
    model="small",
    **calculator_options,
):
    """Relax low-index slabs and calculate their surface energies in eV/A^2.

    gamma = (E_slab - N * E_bulk) / (2 * A), where A is the area of one face.
    """
    try:
        from ase.build import bulk, surface
        from ase.optimize import FIRE
    except ImportError as error:
        raise RuntimeError("Surface-energy calculations require ASE; install it with 'python -m pip install ase'.") from error

    material_name, symbol = _normalize_material(material)
    crystal_structure = _normalize_structure(lattice_structure)
    if crystal_structure not in SURFACE_ORIENTATIONS:
        raise ValueError(f"No surface-orientation defaults are configured for {lattice_structure!r}.")
    if not isinstance(layers, int) or layers < 2:
        raise ValueError("Surface slabs require at least two atomic layers.")
    if vacuum <= 0.0 or fmax <= 0.0 or max_steps <= 0:
        raise ValueError("Vacuum, force tolerance, and optimizer steps must be positive.")

    selected_orientations = orientations or SURFACE_ORIENTATIONS[crystal_structure]
    if calculator is None:
        calculator = create_calculator(
            material_name,
            method,
            model=model,
            espresso_kpts=(8, 8, 8),
            **calculator_options,
        )

    if lattice_parameter is None:
        lattice_parameter = REFERENCE_LATTICE_PARAMETERS[symbol]
    bulk_atoms = bulk(symbol, crystal_structure, a=float(lattice_parameter), cubic=True)
    if bulk_energy_per_atom is None:
        bulk_atoms.calc = calculator
        bulk_energy_per_atom = bulk_atoms.get_potential_energy() / len(bulk_atoms)

    method_name = str(method).strip().upper().replace("_", "-")
    if method_name == "DFT" and hasattr(calculator, "set"):
        calculator.set(kpts=(8, 8, 1))

    surfaces = []
    for indices in selected_orientations:
        slab = surface(
            bulk_atoms,
            indices,
            layers=layers,
            vacuum=vacuum,
            periodic=True,
        )
        slab.pbc = (True, True, True)
        slab.calc = calculator
        optimizer = FIRE(slab, logfile=None)
        converged = optimizer.run(fmax=fmax, steps=max_steps)
        slab_energy = slab.get_potential_energy()
        cell = slab.cell.array
        area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
        energy = surface_energy_from_slab(
            slab_energy,
            bulk_energy_per_atom,
            len(slab),
            area,
        )
        surfaces.append({
            "orientation": "(" + "".join(str(index) for index in indices) + ")",
            "miller_indices": tuple(indices),
            "energy_eV_A2": energy,
            "slab_energy_eV": float(slab_energy),
            "bulk_energy_per_atom_eV": float(bulk_energy_per_atom),
            "atom_count": len(slab),
            "area_A2": area,
            "converged": bool(converged),
        })

    return {
        "material": material_name,
        "symbol": symbol,
        "lattice_structure": lattice_structure,
        "method": method_name,
        "surfaces": surfaces,
    }


def calculate_surface_energy(material, lattice_structure, method, **options):
    """Calculate the (111) surface energy for compatibility with the old API."""
    crystal_structure = _normalize_structure(lattice_structure)
    preferred_orientation = (1, 1, 1)
    result = compute_surface_energies(
        material,
        lattice_structure,
        method,
        orientations=[preferred_orientation],
        **options,
    )
    return result["surfaces"][0]["energy_eV_A2"]
import unittest

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from src.analysis.surface_energy import (
    compute_surface_energies,
    surface_energy_from_slab,
)


class FixedSurfaceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        cell = atoms.cell.array
        area = np.linalg.norm(np.cross(cell[0], cell[1]))
        self.results = {
            "energy": -3.0 * len(atoms) + 0.3 * area,
            "forces": np.zeros((len(atoms), 3)),
        }


class TestSurfaceEnergy(unittest.TestCase):
    def test_surface_energy_uses_two_faces_and_surface_area(self):
        result = surface_energy_from_slab(
            slab_energy=-9.0,
            bulk_energy_per_atom=-1.0,
            atom_count=8,
            surface_area=2.0,
        )
        self.assertAlmostEqual(result, -0.25)

    def test_surface_energy_rejects_zero_area(self):
        with self.assertRaises(ValueError):
            surface_energy_from_slab(0.0, 0.0, 4, 0.0)

    def test_surface_calculation_builds_and_relaxes_requested_facet(self):
        result = compute_surface_energies(
            "Aluminum",
            "FCC",
            "MACE-MP",
            calculator=FixedSurfaceCalculator(),
            bulk_energy_per_atom=-3.0,
            lattice_parameter=4.05,
            orientations=[(1, 0, 0)],
            layers=4,
        )
        surface = result["surfaces"][0]
        self.assertEqual(surface["orientation"], "(100)")
        self.assertAlmostEqual(surface["energy_eV_A2"], 0.15)
        self.assertTrue(surface["converged"])


if __name__ == "__main__":
    unittest.main()
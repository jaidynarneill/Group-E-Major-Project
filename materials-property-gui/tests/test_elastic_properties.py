import unittest
from unittest.mock import patch

import numpy as np

from src.analysis.elastic_properties import (
    MEAM_CATALOG,
    MACE_MODEL_OPTIONS,
    birch_murnaghan,
    create_calculator,
    derive_cubic_properties,
    fit_linear_response,
)


class TestElasticProperties(unittest.TestCase):
    def test_meam_catalog_contains_cubic_material_records(self):
        self.assertEqual(MEAM_CATALOG["Al"]["library_element"], "Al")
        self.assertEqual(MEAM_CATALOG["Cu"]["lattice"], "fcc")
        self.assertEqual(MEAM_CATALOG["Si"]["lattice"], "diamond")
        self.assertIn("Bi", MEAM_CATALOG)
        self.assertNotIn("Mg", MEAM_CATALOG)

    def test_birch_murnaghan_returns_equilibrium_energy_at_v0(self):
        energy = birch_murnaghan(16.0, -3.0, 16.0, 0.5, 4.0)
        self.assertAlmostEqual(energy, -3.0)

    def test_linear_response_recovers_slope_and_fit_quality(self):
        strain = np.array([-0.01, 0.0, 0.01])
        stress = 120.0 * strain + 0.25
        slope, intercept, r_squared = fit_linear_response(strain, stress)
        self.assertAlmostEqual(slope, 120.0)
        self.assertAlmostEqual(intercept, 0.25)
        self.assertAlmostEqual(r_squared, 1.0)

    def test_derived_cubic_properties(self):
        properties = derive_cubic_properties(150.0, 75.0, 50.0)
        self.assertAlmostEqual(properties["B_Cij"], 100.0)
        self.assertGreater(properties["G_H"], 0.0)
        self.assertGreater(properties["E"], 0.0)
        self.assertGreater(properties["A"], 0.0)

    def test_mace_variants_select_the_requested_checkpoint(self):
        for method, model in MACE_MODEL_OPTIONS.items():
            with self.subTest(method=method), patch("mace.calculators.mace_mp") as mace_mp:
                create_calculator("Aluminum", method)
                mace_mp.assert_called_once_with(model=model, default_dtype="float64")

    def test_dft_cannot_fall_back_to_a_local_calculator(self):
        with self.assertRaisesRegex(RuntimeError, "remote VASP Slurm workflow"):
            create_calculator("Aluminum", "DFT")


if __name__ == "__main__":
    unittest.main()
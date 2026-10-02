import unittest
from unittest.mock import patch

import numpy as np

from src.analysis.elastic_properties import (
    MEAM_CATALOG,
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

    def test_dft_calculator_uses_bundled_sssp_pseudopotential_and_cutoffs(self):
        with patch("src.analysis.elastic_properties.shutil.which", return_value="pw.x"):
            calculator = create_calculator("Aluminum", "DFT")

        self.assertEqual(
            calculator.profile.pseudo_dir.rsplit("\\", 1)[-1],
            "library",
        )
        self.assertEqual(
            calculator.parameters["pseudopotentials"]["Al"],
            "Al.us.pbe.z_3.ld1.psl.v1.0.0-low.upf",
        )
        self.assertEqual(calculator.parameters["input_data"]["system"]["ecutwfc"], 30.0)
        self.assertEqual(calculator.parameters["input_data"]["system"]["ecutrho"], 60.0)


if __name__ == "__main__":
    unittest.main()
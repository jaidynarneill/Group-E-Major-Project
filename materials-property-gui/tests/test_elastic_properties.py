import unittest
from src.analysis.elastic_properties import calculate_elastic_properties
import unittest

import numpy as np

from src.analysis.elastic_properties import (
    birch_murnaghan,
    derive_cubic_properties,
    fit_linear_response,
)


class TestElasticProperties(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
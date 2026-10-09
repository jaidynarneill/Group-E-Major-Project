import tempfile
import unittest

import numpy as np

from src.analysis.result_cache import CalculationResultCache


class TestCalculationResultCache(unittest.TestCase):
    def test_round_trip_preserves_plot_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = CalculationResultCache(directory)
            result = {
                "eos_data": {"volume": np.array([10.0, 11.0])},
                "surface_energy_data": {"surfaces": []},
            }

            cache.save("Aluminum", "FCC", "MACE-MP Small", result)
            loaded = cache.load("Aluminum", "FCC", "MACE-MP Small")

        np.testing.assert_array_equal(loaded["eos_data"]["volume"], [10.0, 11.0])

    def test_cache_entries_are_isolated_by_material_and_model(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = CalculationResultCache(directory)
            cache.save("Aluminum", "FCC", "MACE-MP Small", {"value": 1})

            self.assertIsNone(cache.load("Copper", "FCC", "MACE-MP Small"))
            self.assertIsNone(cache.load("Aluminum", "FCC", "MACE-MP Medium"))


if __name__ == "__main__":
    unittest.main()
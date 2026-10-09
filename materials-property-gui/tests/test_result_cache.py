import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.analysis.result_cache import CalculationResultCache, _cache_directory


class TestCalculationResultCache(unittest.TestCase):
    def test_packaged_app_cache_is_next_to_executable(self):
        executable = Path(tempfile.gettempdir()) / "MaterialsPropertyGUI.exe"
        with (
            patch.dict(os.environ, {"MATERIALS_PROPERTY_CACHE": ""}),
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "executable", str(executable)),
        ):
            self.assertEqual(_cache_directory(), executable.parent / "calculations")

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
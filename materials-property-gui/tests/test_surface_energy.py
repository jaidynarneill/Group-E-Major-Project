import unittest
from src.analysis.surface_energy import calculate_surface_energy

class TestSurfaceEnergy(unittest.TestCase):

    def test_surface_energy_copper_meam(self):
        # Test surface energy calculation for copper using MEAM
        result = calculate_surface_energy(material='copper', method='MEAM', lattice_structure='FCC')
        expected = 1.61  # Example expected value in J/m^2
        self.assertAlmostEqual(result, expected, places=2)

    def test_surface_energy_aluminum_meam(self):
        # Test surface energy calculation for aluminum using MEAM
        result = calculate_surface_energy(material='aluminum', method='MEAM', lattice_structure='FCC')
        expected = 1.00  # Example expected value in J/m^2
        self.assertAlmostEqual(result, expected, places=2)

    def test_surface_energy_silicon_dft(self):
        # Test surface energy calculation for silicon using DFT
        result = calculate_surface_energy(material='silicon', method='DFT', lattice_structure='Diamond')
        expected = 0.90  # Example expected value in J/m^2
        self.assertAlmostEqual(result, expected, places=2)

    def test_surface_energy_copper_mace_mp(self):
        # Test surface energy calculation for copper using MACE-MP
        result = calculate_surface_energy(material='copper', method='MACE-MP', lattice_structure='FCC')
        expected = 1.70  # Example expected value in J/m^2
        self.assertAlmostEqual(result, expected, places=2)

if __name__ == '__main__':
    unittest.main()
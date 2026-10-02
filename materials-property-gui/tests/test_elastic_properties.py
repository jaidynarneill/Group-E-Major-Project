import unittest
from src.analysis.elastic_properties import calculate_elastic_properties

class TestElasticProperties(unittest.TestCase):

    def test_elastic_properties_copper_meam(self):
        material = 'copper'
        method = 'MEAM'
        expected_results = {
            'C11': 130.0,  # Example expected value
            'C12': 80.0,   # Example expected value
            'C44': 50.0    # Example expected value
        }
        results = calculate_elastic_properties(material, method)
        self.assertAlmostEqual(results['C11'], expected_results['C11'], delta=1e-2)
        self.assertAlmostEqual(results['C12'], expected_results['C12'], delta=1e-2)
        self.assertAlmostEqual(results['C44'], expected_results['C44'], delta=1e-2)

    def test_elastic_properties_aluminum_dft(self):
        material = 'aluminum'
        method = 'DFT'
        expected_results = {
            'C11': 110.0,  # Example expected value
            'C12': 60.0,   # Example expected value
            'C44': 30.0    # Example expected value
        }
        results = calculate_elastic_properties(material, method)
        self.assertAlmostEqual(results['C11'], expected_results['C11'], delta=1e-2)
        self.assertAlmostEqual(results['C12'], expected_results['C12'], delta=1e-2)
        self.assertAlmostEqual(results['C44'], expected_results['C44'], delta=1e-2)

    def test_elastic_properties_silicon_mace_mp(self):
        material = 'silicon'
        method = 'MACE-MP'
        expected_results = {
            'C11': 160.0,  # Example expected value
            'C12': 70.0,   # Example expected value
            'C44': 45.0    # Example expected value
        }
        results = calculate_elastic_properties(material, method)
        self.assertAlmostEqual(results['C11'], expected_results['C11'], delta=1e-2)
        self.assertAlmostEqual(results['C12'], expected_results['C12'], delta=1e-2)
        self.assertAlmostEqual(results['C44'], expected_results['C44'], delta=1e-2)

if __name__ == '__main__':
    unittest.main()
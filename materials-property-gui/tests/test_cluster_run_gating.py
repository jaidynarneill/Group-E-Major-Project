import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from src.analysis.elastic_properties import MEAM_CATALOG
from src.ui.plot_panel import (
    CalculationWorker,
    PlotPanel,
    _surface_energy_warning,
    _configuration_plot_caption,
    configuration_compatibility_errors,
    configurations_need_cluster_login,
)
from src.ui.cluster_connection_dialog import (
    CREDENTIAL_ACCOUNT,
    CREDENTIAL_SERVICE,
    _clear_saved_credentials,
    _load_saved_credentials,
    _save_credentials,
)


class TestClusterRunGating(unittest.TestCase):
    def test_plot_caption_uses_selected_configuration_method(self):
        self.assertEqual(
            _configuration_plot_caption("Run 2: Aluminum / FCC / MEAM"),
            "Run 2 | Aluminum | FCC | MEAM",
        )

    def test_dropdown_lists_only_materials_with_meam_records(self):
        self.assertEqual(
            PlotPanel.MATERIALS,
            list(dict.fromkeys(record["name"] for record in MEAM_CATALOG.values())),
        )
        self.assertIn("Aluminum", PlotPanel.MATERIALS)
        self.assertNotIn("Cobalt", PlotPanel.MATERIALS)
        self.assertNotIn("Oganesson", PlotPanel.MATERIALS)
        self.assertIn("MEAM", PlotPanel.METHODS)
        self.assertIn("DFT", PlotPanel.METHODS)
        self.assertIn("MACE-MP Small", PlotPanel.METHODS)

    def test_lattice_defaults_follow_material_and_meam_structure(self):
        self.assertEqual(PlotPanel.lattices_for_method("Aluminum", "MEAM"), ["FCC"])
        self.assertEqual(
            PlotPanel.lattices_for_method("Aluminum", "DFT"),
            PlotPanel.LATTICES["Aluminum"],
        )
        self.assertEqual(PlotPanel.default_lattice_for_material("Silicon"), "Diamond cubic")

    def test_meam_incompatibility_is_reported_without_hiding_the_method(self):
        errors = configuration_compatibility_errors([(1, "Cobalt", "FCC", "MEAM")])

        self.assertEqual(len(errors), 1)
        self.assertIn("MEAM is incompatible with Cobalt", errors[0])

    def test_dft_cobalt_is_compatible_but_unknown_starting_cells_are_rejected(self):
        self.assertEqual(
            configuration_compatibility_errors([(1, "Co", "FCC", "DFT")]),
            [],
        )
        errors = configuration_compatibility_errors([(1, "Og", "FCC", "DFT")])
        self.assertTrue(any("no reference cell" in error for error in errors))

    def test_unsupported_lattice_is_reported_when_run_is_requested(self):
        errors = configuration_compatibility_errors([(1, "Cobalt", "HCP", "DFT")])

        self.assertEqual(len(errors), 1)
        self.assertIn("Cubic elasticity fits", errors[0])

    def test_credentials_are_remembered_through_the_os_keyring(self):
        entries = {}
        keyring = ModuleType("keyring")
        keyring.get_password = lambda service, account: entries.get((service, account))
        keyring.set_password = lambda service, account, value: entries.__setitem__((service, account), value)
        keyring.delete_password = lambda service, account: entries.pop((service, account), None)

        with patch.dict("sys.modules", {"keyring": keyring}):
            _save_credentials("student", "secret")
            loaded = _load_saved_credentials()

        self.assertEqual(loaded, ("student", "secret"))
        self.assertEqual(len(entries), 1)
        self.assertIn((CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT), entries)

    def test_invalid_saved_credential_data_is_ignored(self):
        keyring = ModuleType("keyring")
        keyring.get_password = lambda service, account: "not valid credential json"
        with patch.dict("sys.modules", {"keyring": keyring}):
            self.assertIsNone(_load_saved_credentials())

    def test_unchecking_remember_me_clears_saved_credentials(self):
        entries = {(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT): '{"username":"student","password":"secret"}'}
        keyring = ModuleType("keyring")
        keyring.get_password = lambda service, account: entries.get((service, account))
        keyring.delete_password = lambda service, account: entries.pop((service, account), None)

        with patch.dict("sys.modules", {"keyring": keyring}):
            _clear_saved_credentials()
            loaded = _load_saved_credentials()

        self.assertIsNone(loaded)
        self.assertEqual(entries, {})

    def test_local_only_run_does_not_require_cluster_login(self):
        configurations = [
            (1, "Copper", "FCC", "MEAM"),
            (3, "Silicon", "Diamond cubic", "MACE-MP Small"),
        ]
        self.assertFalse(configurations_need_cluster_login(configurations))

    def test_enabled_dft_run_requires_cluster_login(self):
        configurations = [
            (2, "Sodium", "BCC", "DFT"),
        ]
        self.assertTrue(configurations_need_cluster_login(configurations))

    def test_cached_dft_run_does_not_require_cluster_login(self):
        cache = Mock()
        cache.load.return_value = {"method": "DFT"}

        self.assertFalse(
            configurations_need_cluster_login(
                [(2, "Sodium", "BCC", "DFT")],
                result_cache=cache,
            )
        )
        cache.load.assert_called_once_with("Sodium", "BCC", "DFT")

    def test_uncached_dft_run_still_requires_cluster_login(self):
        cache = Mock()
        cache.load.return_value = None

        self.assertTrue(
            configurations_need_cluster_login(
                [(2, "Sodium", "BCC", "DFT")],
                result_cache=cache,
            )
        )

    def test_dft_label_matching_ignores_case_and_whitespace(self):
        self.assertTrue(
            configurations_need_cluster_login([(1, "Sodium", "BCC", " dft ")])
        )

    def test_dft_worker_uses_remote_vasp_and_drops_credentials(self):
        worker = CalculationWorker(
            [(2, "Sodium", "BCC", "DFT")],
            cluster_credentials=("student", "secret"),
        )
        worker.result_cache.load = Mock(return_value=None)
        worker.result_cache.save = Mock()
        completed = []
        worker.completed.connect(lambda results, errors: completed.append((results, errors)))
        result = {"material": "Sodium", "method": "DFT"}

        with (
            patch("src.dft.vasp_workflow.calculate_remote_dft_report", return_value=result) as remote_run,
            patch("src.ui.plot_panel.create_calculator") as local_calculator,
        ):
            worker.run()

        remote_run.assert_called_once()
        self.assertEqual(remote_run.call_args.args, ("Sodium", "BCC", "student", "secret"))
        self.assertTrue(callable(remote_run.call_args.kwargs["progress"]))
        local_calculator.assert_not_called()
        self.assertEqual(completed[0], ([("Run 2: Sodium / BCC / DFT", result)], []))
        self.assertIsNone(worker.cluster_credentials)

    def test_dft_worker_keeps_elastic_results_when_surface_phase_fails(self):
        worker = CalculationWorker(
            [(1, "Sodium", "BCC", "DFT")],
            cluster_credentials=("student", "secret"),
        )
        worker.result_cache.load = Mock(return_value=None)
        worker.result_cache.save = Mock()
        completed = []
        worker.completed.connect(lambda results, errors: completed.append((results, errors)))
        result = {
            "material": "Sodium",
            "method": "DFT",
            "elastic_data": {"sigma_xy_GPa": [1.0, 2.0]},
            "surface_energy_data": {"surfaces": []},
            "surface_energy_error": "surface job failed for secret",
        }

        with patch("src.dft.vasp_workflow.calculate_remote_dft_report", return_value=result):
            worker.run()

        self.assertEqual(completed[0][0], [("Run 1: Sodium / BCC / DFT", result)])
        self.assertEqual(result["elastic_data"]["sigma_xy_GPa"], [1.0, 2.0])
        self.assertEqual(result["surface_energy_error"], "surface job failed for [redacted]")
        self.assertEqual(completed[0][1], [])
        worker.result_cache.save.assert_called_once_with("Sodium", "BCC", "DFT", result)

        cached_worker = CalculationWorker([(1, "Sodium", "BCC", "DFT")])
        cached_worker.result_cache.load = Mock(return_value=result)
        cached_completed = []
        cached_worker.completed.connect(
            lambda results, errors: cached_completed.append((results, errors))
        )
        cached_worker.run()

        self.assertEqual(cached_completed[0][0], [("Run 1: Sodium / BCC / DFT", result)])
        self.assertEqual(cached_completed[0][1], [])

    def test_cached_partial_surface_data_is_recognized_without_failure_metadata(self):
        warning = _surface_energy_warning(
            "Aluminum",
            "FCC",
            {
                "surface_energy_data": {
                    "surfaces": [{"orientation": "(100)"}],
                },
            },
        )

        self.assertEqual(
            warning,
            "Incomplete surface-energy data; missing (110), (111).",
        )

    def test_partial_cached_dft_report_loads_without_remote_retry(self):
        worker = CalculationWorker([(1, "Sodium", "BCC", "DFT")])
        cached_result = {
            "material": "Sodium",
            "lattice_structure": "BCC",
            "method": "DFT",
            "elastic_data": {"sigma_xy_GPa": [1.0, 2.0]},
            "surface_energy_data": {"surfaces": []},
        }
        worker.result_cache.load = Mock(return_value=cached_result)
        completed = []
        worker.completed.connect(lambda results, errors: completed.append((results, errors)))

        with patch(
            "src.dft.vasp_workflow.calculate_remote_dft_report",
        ) as remote_run:
            worker.run()

        remote_run.assert_not_called()
        self.assertEqual(completed[0], ([("Run 1: Sodium / BCC / DFT", cached_result)], []))

    def test_dft_worker_redacts_credentials_from_remote_errors(self):
        worker = CalculationWorker(
            [(1, "Sodium", "BCC", "DFT")],
            cluster_credentials=("student", "secret"),
        )
        worker.result_cache.load = Mock(return_value=None)
        completed = []
        worker.completed.connect(lambda results, errors: completed.append((results, errors)))

        with patch(
            "src.dft.vasp_workflow.calculate_remote_dft_report",
            side_effect=RuntimeError("SSH rejected student with password secret"),
        ):
            worker.run()

        error_report = "\n".join(completed[0][1])
        self.assertNotIn("student", error_report)
        self.assertNotIn("secret", error_report)
        self.assertIn("[redacted]", error_report)


if __name__ == "__main__":
    unittest.main()
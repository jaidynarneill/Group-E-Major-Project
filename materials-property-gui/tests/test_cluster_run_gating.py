import unittest
from unittest.mock import Mock, patch

from src.ui.plot_panel import CalculationWorker, configurations_need_cluster_login


class TestClusterRunGating(unittest.TestCase):
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
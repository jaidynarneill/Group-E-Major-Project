import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
from ase.build import bulk

from src.analysis.elastic_properties import EV_A3_TO_GPA, birch_murnaghan
from src.dft.vasp_workflow import (
    _assemble_report,
    _job_accounting_command,
    _job_log_tail_command,
    _remote_material_directory,
    _stage_phase,
    _validate_remote_environment,
    _wait_for_job,
    build_bulk_relaxation_task,
    build_eos_tasks,
    build_followup_tasks,
    calculate_remote_dft_report,
    _find_remote_material_directory,
    _inspect_remote_run,
)
from src.dft.remote_vasp import REMOTE_SCRATCH_DIRECTORY, VASP_MODULE
from src.dft.vasp_workflow import VASP_MPI_TASKS, VASP_WORKFLOW_VERSION


class TestVaspWorkflow(unittest.TestCase):
    def test_surface_amin_setting_invalidates_previous_dft_cache(self):
        self.assertEqual(VASP_WORKFLOW_VERSION, 8)
        self.assertEqual(VASP_MPI_TASKS, 16)

    def test_failed_job_diagnostics_include_slurm_resource_accounting(self):
        command = _job_accounting_command("61084245")

        self.assertIn("--format=State,ExitCode,MaxRSS,ReqMem,Elapsed", command)
        self.assertIn("--jobs=61084245", command)

        client = Mock()
        with patch(
            "src.dft.vasp_workflow._remote_command",
            side_effect=[
                ("FAILED", "", 0),
                ("surface_100 vasp.log and OUTCAR tail", "", 0),
                ("FAILED|139:0|3G|4G|00:01:00", "", 0),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "FAILED\\|139:0\\|3G"):
                _wait_for_job(client, "/scratch/job/followup", "61084245", Mock(), 0, 30)

    def test_material_directory_is_the_flat_run_root(self):
        material_directory = _remote_material_directory("Aluminum")

        self.assertTrue(material_directory.endswith("/Aluminum-Dft"))

    def test_material_directory_lookup_ignores_folder_name_case(self):
        sftp = Mock()
        sftp.listdir.return_value = ["Copper-Dft", "aluminum-dft"]

        self.assertEqual(
            _find_remote_material_directory(sftp, "Aluminum"),
            f"{REMOTE_SCRATCH_DIRECTORY}/aluminum-dft",
        )

    def test_job_poller_reports_state_without_streaming_logs(self):
        client = Mock()
        progress = Mock()
        with (
            patch(
                "src.dft.vasp_workflow._remote_command",
                side_effect=[
                    ("RUNNING", "", 0),
                    ("COMPLETED", "", 0),
                ],
            ),
            patch("src.dft.vasp_workflow.time.sleep"),
        ):
            _wait_for_job(client, "/scratch/job/eos", "98765", progress, 0, 30)

        self.assertEqual(
            [call.args[0] for call in progress.call_args_list],
            ["Slurm job 98765: RUNNING...", "Slurm job 98765: COMPLETED"],
        )

    def test_log_tail_reads_task_vasp_and_persistent_slurm_outputs(self):
        command = _job_log_tail_command("/scratch/job/eos", "98765")

        self.assertIn("/scratch/job/eos/tasks/*/vasp.log", command)
        self.assertIn("/scratch/job/eos/tasks/*/OUTCAR", command)
        self.assertIn("/scratch/job/eos/slurm-*-98765.out", command)
        self.assertIn("/scratch/job/eos/slurm-*-98765.err", command)
        self.assertIn("tail -n 12", command)

    def test_cobalt_relaxation_and_eos_inputs_use_logical_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relaxation_task = build_bulk_relaxation_task("Co", "FCC", root / "relax")
            _, _, _, eos_tasks = build_eos_tasks(
                "Co",
                "FCC",
                root / "eos",
                reference_lattice=3.5133144702592642,
            )

            relaxation_directory = root / "relax" / relaxation_task["name"]
            relaxation_poscar = (relaxation_directory / "POSCAR").read_text(encoding="ascii")
            relaxation_kpoints = (relaxation_directory / "KPOINTS").read_text(encoding="ascii")
            relaxation_incar = (relaxation_directory / "INCAR.template").read_text(encoding="ascii")
            eos_incar = (root / "eos" / eos_tasks[0]["name"] / "INCAR.template").read_text(encoding="ascii")

        self.assertEqual([line.strip() for line in relaxation_poscar.splitlines()[5:7]], ["Co", "4"])
        self.assertIn("Gamma\n7 7 7\n", relaxation_kpoints)
        self.assertIn("ISIF = 3", relaxation_incar)
        self.assertIn("NSW = 99", relaxation_incar)
        self.assertIn("LREAL = Auto", relaxation_incar)
        self.assertNotIn("AMIN", relaxation_incar)
        self.assertIn("ISPIN = 2", relaxation_incar)
        self.assertIn("MAGMOM = 4*0.6", relaxation_incar)
        self.assertIn("ISIF = 2", eos_incar)
        self.assertIn("IBRION = -1", eos_incar)
        self.assertIn("NSW = 0", eos_incar)
        self.assertNotIn("EDIFFG", eos_incar)
        self.assertAlmostEqual(eos_tasks[0]["lattice_parameter"], 3.5133144702592642 * 0.97)

    def test_aluminum_surface_relaxation_uses_same_projector_mode_as_working_shear(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = build_followup_tasks("Al", "fcc", 4.05, directory)
            surface_directory = Path(directory) / "surface_100"
            incar = (surface_directory / "INCAR.template").read_text(encoding="ascii")
            kpoints = (surface_directory / "KPOINTS").read_text(encoding="ascii")

        surface_task = next(task for task in tasks if task["name"] == "surface_100")
        self.assertEqual(surface_task["atom_count"], 32)
        self.assertIn("LREAL = .FALSE.", incar)
        self.assertIn("AMIN = 0.01", incar)
        self.assertIn("KPAR = 4", incar)
        self.assertIn("ISIF = 2", incar)
        self.assertIn("Gamma\n8 8 1\n", kpoints)

    def test_remote_preflight_requires_vasp_and_potgen(self):
        with patch(
            "src.dft.vasp_workflow._remote_command",
            return_value=("", "", 0),
        ) as remote_command:
            _validate_remote_environment(Mock(), "Al")

        command = remote_command.call_args.args[1]
        self.assertIn(f"module load {VASP_MODULE}", command)
        self.assertIn("command -v potgen", command)
        self.assertNotIn("POTCAR_SOURCE", command)

    def test_phase_stages_task_inputs_submits_and_downloads_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            local_phase = Path(directory)
            task_directory = local_phase / "eos_00"
            task_directory.mkdir()
            for filename in ("POSCAR", "KPOINTS", "INCAR.template"):
                (task_directory / filename).write_text(filename, encoding="ascii")
            sftp = Mock()
            client = Mock()
            progress = Mock()
            command_outputs = [
                ("", "", 0),
                ("", "", 0),
                ("", "", 0),
                ("98765;cluster", "", 0),
            ]

            with (
                patch("src.dft.vasp_workflow._remote_command", side_effect=command_outputs) as remote_command,
                patch("src.dft.vasp_workflow._wait_for_job") as wait_for_job,
                patch("src.dft.vasp_workflow._read_vasp_result", return_value={"energy": -1.25}) as read_result,
            ):
                result = _stage_phase(
                    sftp,
                    client,
                    "/scratch/job",
                    "eos",
                    [{"name": "eos_00", "kind": "eos"}],
                    "Na",
                    local_phase,
                    "dft-na-eos",
                    progress,
                    0.01,
                    60,
                )

        self.assertEqual(result, {"eos_00": {"energy": -1.25}})
        self.assertEqual(sftp.put.call_count, 4)
        sftp.putfo.assert_called_once()
        sftp.get.assert_called_once_with(
            "/scratch/job/eos/tasks/eos_00/OUTCAR",
            str(task_directory / "OUTCAR"),
        )
        wait_for_job.assert_called_once_with(client, "/scratch/job/eos", "98765", progress, 0.01, 60)
        read_result.assert_called_once_with(task_directory / "OUTCAR")
        self.assertIn("sbatch --parsable", remote_command.call_args_list[-1].args[1])

    def test_na_bcc_task_generation_writes_full_report_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, symbol, structure, eos_tasks = build_eos_tasks("Sodium", "BCC", root / "eos")
            followup_tasks = build_followup_tasks(symbol, structure, 4.2, root / "followup")

            self.assertEqual(len(eos_tasks), 13)
            self.assertEqual(len(followup_tasks), 15)
            for task in (eos_tasks[0], followup_tasks[-1]):
                task_directory = root / ("eos" if task in eos_tasks else "followup") / task["name"]
                self.assertTrue((task_directory / "POSCAR").is_file())
                self.assertTrue((task_directory / "KPOINTS").is_file())
                self.assertTrue((task_directory / "INCAR.template").is_file())
            surfaces = [task for task in followup_tasks if task["kind"] == "surface"]
            self.assertEqual([task["atom_count"] for task in surfaces], [16, 16, 16])

    def test_report_assembly_matches_gui_result_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            _, symbol, structure, eos_tasks = build_eos_tasks("Sodium", "BCC", Path(directory) / "eos")
            atom_count = len(bulk(symbol, structure, a=eos_tasks[0]["lattice_parameter"], cubic=True))
            equilibrium_volume = 78.0
            equilibrium_energy = -2.3 * atom_count
            eos_outputs = {}
            for task in eos_tasks:
                atoms = bulk(symbol, structure, a=task["lattice_parameter"], cubic=True)
                eos_outputs[task["name"]] = {
                    "energy": float(birch_murnaghan(
                        atoms.get_volume(), equilibrium_energy, equilibrium_volume, 0.4, 4.0
                    )),
                }
            equilibrium_lattice = equilibrium_volume ** (1.0 / 3.0)
            followup_tasks = build_followup_tasks(symbol, structure, equilibrium_lattice, Path(directory) / "followup")
            followup_outputs = {}
            expected_c11, expected_c12, expected_c44 = 180.0, 90.0, 45.0
            for task in followup_tasks:
                if task["kind"] == "normal_stress":
                    strain = task["strain"]
                    stress = np.array([
                        expected_c11 * strain / EV_A3_TO_GPA,
                        expected_c12 * strain / EV_A3_TO_GPA,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ])
                    followup_outputs[task["name"]] = {"stress": stress}
                elif task["kind"] == "shear_stress":
                    stress = np.zeros(6)
                    stress[5] = expected_c44 * task["strain"] / EV_A3_TO_GPA
                    followup_outputs[task["name"]] = {"stress": stress}
                else:
                    followup_outputs[task["name"]] = {
                        "energy": equilibrium_energy / atom_count * task["atom_count"]
                        + 2 * task["area_A2"] * 0.02,
                        "forces": np.zeros((task["atom_count"], 3)),
                    }

            report = _assemble_report(
                "Sodium",
                "Na",
                "BCC",
                eos_tasks,
                eos_outputs,
                followup_tasks,
                followup_outputs,
            )

        self.assertEqual(report["method"], "DFT")
        self.assertAlmostEqual(report["V0"], equilibrium_volume, places=4)
        self.assertAlmostEqual(report["C11"], expected_c11, places=5)
        self.assertAlmostEqual(report["C12"], expected_c12, places=5)
        self.assertAlmostEqual(report["C44"], expected_c44, places=5)
        self.assertEqual(len(report["surface_energy_data"]["surfaces"]), 3)
        self.assertTrue(all(item["converged"] for item in report["surface_energy_data"]["surfaces"]))

        elastic_outputs = {
            task["name"]: followup_outputs[task["name"]]
            for task in followup_tasks
            if task["kind"] != "surface"
        }
        elastic_report = _assemble_report(
            "Sodium",
            "Na",
            "BCC",
            eos_tasks,
            eos_outputs,
            followup_tasks,
            elastic_outputs,
        )
        self.assertAlmostEqual(elastic_report["C11"], expected_c11, places=5)
        self.assertAlmostEqual(elastic_report["C12"], expected_c12, places=5)
        self.assertAlmostEqual(elastic_report["C44"], expected_c44, places=5)
        self.assertEqual(elastic_report["surface_energy_data"]["surfaces"], [])

    def test_surface_phase_failure_preserves_elastic_report(self):
        client = Mock()
        client.open_sftp.return_value.listdir.return_value = []
        eos_tasks = [{"name": "eos_00", "lattice_parameter": 4.0}]
        followup_tasks = [
            {"name": "normal_00", "kind": "normal_stress"},
            {"name": "shear_00", "kind": "shear_stress"},
            {"name": "surface_100", "kind": "surface"},
        ]
        elastic_outputs = {
            "normal_00": {"stress": np.ones(6)},
            "shear_00": {"stress": np.ones(6)},
        }
        report = {
            "method": "DFT",
            "elastic_data": {"sigma_xy_GPa": np.ones(3)},
            "surface_energy_data": {"surfaces": []},
        }
        stage_results = [
            {"bulk_relax": {"volume_A3": 64.0}},
            {"eos_00": {"energy": -1.0}},
            elastic_outputs,
            RuntimeError("surface Slurm job failed"),
        ]

        with (
            patch("src.dft.vasp_workflow._connect", return_value=client),
            patch("src.dft.vasp_workflow._find_remote_material_directory", return_value=None),
            patch("src.dft.vasp_workflow._validate_remote_environment"),
            patch("src.dft.vasp_workflow._remote_command"),
            patch("src.dft.vasp_workflow.build_bulk_relaxation_task", return_value={"name": "bulk_relax"}),
            patch(
                "src.dft.vasp_workflow.build_eos_tasks",
                return_value=("Aluminum", "Al", "fcc", eos_tasks),
            ),
            patch("src.dft.vasp_workflow._fit_eos", return_value=np.array([-1.0, 64.0, 0.4, 4.0])),
            patch("src.dft.vasp_workflow.build_followup_tasks", return_value=followup_tasks),
            patch("src.dft.vasp_workflow._stage_phase", side_effect=stage_results) as stage_phase,
            patch("src.dft.vasp_workflow._assemble_report", return_value=report) as assemble_report,
        ):
            result = calculate_remote_dft_report("Aluminum", "FCC", "student", "secret")

        self.assertEqual(
            [call.args[3] for call in stage_phase.call_args_list],
            ["bulk-relax", "eos", "elastic", "surfaces"],
        )
        self.assertEqual(assemble_report.call_args.args[-1], elastic_outputs)
        self.assertEqual(result["elastic_data"], report["elastic_data"])
        self.assertEqual(result["surface_energy_data"]["surfaces"], [])
        self.assertEqual(result["surface_energy_error"], "surface Slurm job failed")

    def test_complete_remote_run_is_reused_without_preflight_or_submission(self):
        client = Mock()
        report = {"method": "DFT", "material": "Aluminum", "C11": 100.0}
        reusable_run = {
            "report": report,
            "missing_surfaces": [],
            "remote_run": "/shared/Aluminum-Dft/run-complete",
        }

        with (
            patch("src.dft.vasp_workflow._connect", return_value=client),
            patch(
                "src.dft.vasp_workflow._find_remote_material_directory",
                return_value="/shared/Aluminum-Dft",
            ),
            patch("src.dft.vasp_workflow._find_reusable_remote_run", return_value=reusable_run),
            patch("src.dft.vasp_workflow._validate_remote_environment") as preflight,
            patch("src.dft.vasp_workflow._stage_phase") as stage_phase,
        ):
            result = calculate_remote_dft_report("Aluminum", "FCC", "student", "secret")

        self.assertIs(result, report)
        preflight.assert_not_called()
        stage_phase.assert_not_called()

    def test_elastic_complete_remote_run_resumes_only_missing_surfaces(self):
        client = Mock()
        sftp = client.open_sftp.return_value
        sftp.listdir.return_value = []
        surface_task = {"name": "surface_100", "kind": "surface"}
        followup_tasks = [
            {"name": "normal_00", "kind": "normal_stress"},
            {"name": "shear_00", "kind": "shear_stress"},
            surface_task,
        ]
        eos_tasks = [{"name": "eos_00"}]
        eos_outputs = {"eos_00": {"energy": -1.0}}
        elastic_outputs = {
            "normal_00": {"stress": np.ones(6)},
            "shear_00": {"stress": np.ones(6)},
        }
        reusable_run = {
            "remote_run": "/shared/Aluminum-Dft",
            "report": {"a0": 4.0},
            "eos_tasks": eos_tasks,
            "eos_outputs": eos_outputs,
            "followup_tasks": followup_tasks,
            "followup_outputs": elastic_outputs,
            "missing_surfaces": [surface_task],
        }
        surface_output = {"surface_100": {"energy": -0.5, "forces": np.zeros((1, 3))}}
        report = {"method": "DFT", "surface_energy_data": {"surfaces": []}}

        with (
            patch("src.dft.vasp_workflow._connect", return_value=client),
            patch(
                "src.dft.vasp_workflow._find_remote_material_directory",
                return_value="/shared/Aluminum-Dft",
            ),
            patch("src.dft.vasp_workflow._find_reusable_remote_run", return_value=reusable_run),
            patch("src.dft.vasp_workflow._validate_remote_environment"),
            patch("src.dft.vasp_workflow.build_followup_tasks", return_value=followup_tasks),
            patch("src.dft.vasp_workflow._stage_phase", return_value=surface_output) as stage_phase,
            patch("src.dft.vasp_workflow._assemble_report", return_value=report) as assemble_report,
        ):
            result = calculate_remote_dft_report("Aluminum", "FCC", "student", "secret")

        self.assertEqual(result, report)
        self.assertEqual(stage_phase.call_count, 1)
        self.assertTrue(stage_phase.call_args.args[3].startswith("surfaces-resume-"))
        self.assertEqual(stage_phase.call_args.args[2], "/shared/Aluminum-Dft")
        self.assertEqual(stage_phase.call_args.args[4], [surface_task])
        self.assertEqual(
            assemble_report.call_args.args[-1],
            {**elastic_outputs, **surface_output},
        )

    def test_remote_run_inspection_detects_partial_surface_stage(self):
        sftp = Mock()
        sftp.listdir.return_value = ["bulk-relax", "eos", "elastic", "surfaces"]
        eos_tasks = [{"name": "eos_00", "lattice_parameter": 4.0}]
        followup_tasks = [
            {"name": "normal_00", "kind": "normal_stress"},
            {"name": "shear_00", "kind": "shear_stress"},
            {"name": "surface_100", "kind": "surface"},
            {"name": "surface_110", "kind": "surface"},
        ]
        elastic_outputs = {
            "normal_00": {"stress": np.ones(6)},
            "shear_00": {"stress": np.ones(6)},
        }
        surface_output = {"surface_100": {"energy": -0.5, "forces": np.zeros((1, 3))}}
        report = {"method": "DFT", "a0": 4.0}

        with (
            patch(
                "src.dft.vasp_workflow.build_bulk_relaxation_task",
                return_value={"name": "bulk_relax", "kind": "bulk_relax"},
            ),
            patch(
                "src.dft.vasp_workflow._load_remote_phase",
                side_effect=[
                    ({"bulk_relax": {"volume_A3": 64.0}}, []),
                    ({"eos_00": {"energy": -1.0}}, []),
                    (elastic_outputs, []),
                    (surface_output, []),
                    ({}, [followup_tasks[-1]]),
                ],
            ) as load_phase,
            patch(
                "src.dft.vasp_workflow.build_eos_tasks",
                return_value=("Aluminum", "Al", "fcc", eos_tasks),
            ),
            patch("src.dft.vasp_workflow._fit_eos", return_value=np.array([-1.0, 64.0, 0.4, 4.0])),
            patch("src.dft.vasp_workflow.build_followup_tasks", return_value=followup_tasks),
            patch("src.dft.vasp_workflow._assemble_report", return_value=report),
        ):
            result = _inspect_remote_run(sftp, "/shared/Aluminum-Dft/run-one", "Aluminum", "FCC")

        self.assertEqual(result["missing_surfaces"], [followup_tasks[-1]])
        self.assertEqual(result["followup_outputs"]["surface_100"], surface_output["surface_100"])
        self.assertEqual(
            [call.args[2] for call in load_phase.call_args_list],
            ["bulk-relax", "eos", "elastic", "surfaces", "surfaces"],
        )


if __name__ == "__main__":
    unittest.main()
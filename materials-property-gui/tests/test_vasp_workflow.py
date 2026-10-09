import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase.build import bulk

from src.analysis.elastic_properties import EV_A3_TO_GPA, birch_murnaghan
from src.dft.vasp_workflow import (
    _assemble_report,
    build_eos_tasks,
    build_followup_tasks,
)


class TestVaspWorkflow(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
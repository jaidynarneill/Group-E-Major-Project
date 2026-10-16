import io
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.dft.remote_vasp import (
    MODULEFILE_DIRECTORY,
    REMOTE_SCRATCH_DIRECTORY,
    REMOTE_SLURM_DIRECTORY,
    SSH_HOST,
    VASP_MODULE,
    inspect_cluster,
)


class TestRemoteVaspInspection(unittest.TestCase):
    def test_launcher_starts_with_shebang_and_keeps_logs_outside_job_workspace(self):
        launcher = Path(__file__).parents[1] / "src" / "dft" / "run.sh"
        launcher_bytes = launcher.read_bytes()

        self.assertTrue(launcher_bytes.startswith(b"#!/bin/bash\n"))
        self.assertIn(
            f"#SBATCH --output={REMOTE_SLURM_DIRECTORY}/%x-%j.out".encode("ascii"),
            launcher_bytes,
        )
        self.assertIn(
            f"#SBATCH --error={REMOTE_SLURM_DIRECTORY}/%x-%j.err".encode("ascii"),
            launcher_bytes,
        )
        launcher = launcher_bytes.decode("ascii")
        self.assertIn("module use /projects/lh36/sdwi0002/opt/modulefiles", launcher)
        self.assertIn("module load vasp/6.6.1", launcher)
        self.assertNotIn("module load hpcx", launcher)
        self.assertNotIn("module load hdf5", launcher)
        self.assertNotIn("module load wannier90", launcher)

    def test_inspection_authenticates_and_reports_remote_capabilities(self):
        client = Mock()
        sftp = Mock()
        sftp.listdir.return_value = ["run.sh", "POTCAR"]
        client.open_sftp.return_value = sftp
        stdout = Mock()
        stdout.read.return_value = b"VASP_EXECUTABLE=/opt/vasp/vasp_std"
        stdout.channel.recv_exit_status.return_value = 0
        stderr = Mock()
        stderr.read.return_value = b""
        client.exec_command.return_value = (None, stdout, stderr)

        report = inspect_cluster("student", "secret", client_factory=lambda: client)

        client.connect.assert_called_once_with(
            hostname=SSH_HOST,
            username="student",
            password="secret",
            timeout=15,
            auth_timeout=20,
            banner_timeout=20,
            allow_agent=False,
            look_for_keys=False,
        )
        command = client.exec_command.call_args.args[0]
        self.assertIn(MODULEFILE_DIRECTORY, command)
        self.assertIn(VASP_MODULE, command)
        self.assertEqual(VASP_MODULE, "vasp/6.6.1")
        self.assertNotIn("hpcx", command)
        self.assertNotIn("hdf5", command)
        self.assertNotIn("wannier90", command)
        self.assertIn("command -v potgen", command)
        self.assertLess(command.index("module use"), command.index("module load vasp/6.6.1"))
        self.assertIn(REMOTE_SCRATCH_DIRECTORY, command)
        self.assertEqual(report["slurm_log_directory"], REMOTE_SLURM_DIRECTORY)
        self.assertEqual(report["scratch_entries"], ["POTCAR", "run.sh"])
        self.assertNotIn("secret", repr(report))
        sftp.close.assert_called_once()
        client.close.assert_called_once()

    def test_inspection_requires_username_and_password_without_connecting(self):
        client = Mock()
        with self.assertRaisesRegex(ValueError, "username and password"):
            inspect_cluster("", "", client_factory=lambda: client)
        client.connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
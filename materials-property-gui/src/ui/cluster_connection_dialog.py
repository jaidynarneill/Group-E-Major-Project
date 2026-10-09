from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
)

try:
    from dft.remote_vasp import (
        MODULEFILE_DIRECTORY,
        REMOTE_SCRATCH_DIRECTORY,
        SSH_HOST,
        VASP_MODULE,
        inspect_cluster,
    )
except ModuleNotFoundError as error:
    if error.name != "dft":
        raise
    from src.dft.remote_vasp import (
        MODULEFILE_DIRECTORY,
        REMOTE_SCRATCH_DIRECTORY,
        SSH_HOST,
        VASP_MODULE,
        inspect_cluster,
    )


class ClusterConnectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DFT cluster credentials")
        self.setMinimumWidth(560)
        self.cluster_report = None
        self.credentials = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Cluster username")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Cluster password")
        form.addRow("SSH host", QLabel(SSH_HOST))
        form.addRow("Username", self.username_input)
        form.addRow("Password", self.password_input)
        form.addRow("VASP module", QLabel(VASP_MODULE))
        form.addRow("Module files", QLabel(MODULEFILE_DIRECTORY))
        form.addRow("Scratch staging folder", QLabel(REMOTE_SCRATCH_DIRECTORY))
        layout.addLayout(form)

        self.status_label = QLabel("Credentials are requested only for enabled DFT runs. No password is saved.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumBlockCount(200)
        self.details.setPlaceholderText("Cluster inspection details will appear here.")
        layout.addWidget(self.details)

        buttons = QDialogButtonBox()
        self.connect_button = buttons.addButton("Connect", QDialogButtonBox.AcceptRole)
        self.connect_button.clicked.connect(self.test_connection)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def test_connection(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        self.details.clear()
        self.status_label.setText("Connecting and checking the cluster...")
        self.connect_button.setEnabled(False)
        QApplication.processEvents()
        try:
            report = inspect_cluster(username, password)
        except Exception as error:
            message = str(error)
            for secret in (password, username):
                if secret:
                    message = message.replace(secret, "[redacted]")
            self.status_label.setText("Connection or inspection failed.")
            self.details.setPlainText(message)
        else:
            self.cluster_report = report
            self.credentials = (username, password)
            self.status_label.setText("SSH login succeeded. No jobs were submitted.")
            lines = [
                f"Host: {report['host']}",
                f"Scratch directory: {report['scratch_directory']}",
                "Scratch directory entries:",
                *(f"  {entry}" for entry in report["scratch_entries"]),
                "VASP module and POTCAR inspection output:",
                report["command_output"] or "(no stdout)",
            ]
            if report["command_errors"]:
                lines.extend(("Inspection stderr:", report["command_errors"]))
            lines.append(f"Inspection exit status: {report['command_exit_status']}")
            self.details.setPlainText("\n".join(lines))
            self.accept()
        finally:
            self.password_input.clear()
            self.connect_button.setEnabled(True)

    def take_credentials(self):
        credentials = self.credentials
        self.credentials = None
        return credentials
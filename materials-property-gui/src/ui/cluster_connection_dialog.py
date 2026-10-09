import json

from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

try:
    from dft.remote_vasp import inspect_cluster
except ModuleNotFoundError as error:
    if error.name != "dft":
        raise
    from src.dft.remote_vasp import inspect_cluster


CREDENTIAL_SERVICE = "MaterialsPropertyGUI"
CREDENTIAL_ACCOUNT = "m3.massive.org.au"


def _load_saved_credentials():
    try:
        import keyring

        saved = keyring.get_password(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT)
        credentials = json.loads(saved) if saved else None
    except Exception:
        return None
    if not isinstance(credentials, dict):
        return None
    username = credentials.get("username")
    password = credentials.get("password")
    if not isinstance(username, str) or not isinstance(password, str):
        return None
    return username, password


def _save_credentials(username, password):
    import keyring

    keyring.set_password(
        CREDENTIAL_SERVICE,
        CREDENTIAL_ACCOUNT,
        json.dumps({"username": username, "password": password}),
    )


class ClusterConnectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster login")
        self.setMinimumWidth(460)
        self.credentials = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Cluster username")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Cluster password")
        for field in (self.username_input, self.password_input):
            field.setMinimumHeight(44)
            field.setStyleSheet("font-size: 18px; padding: 6px;")
        form.addRow("Username", self.username_input)
        form.addRow("Password", self.password_input)
        layout.addLayout(form)

        buttons = QDialogButtonBox()
        self.connect_button = buttons.addButton("Connect", QDialogButtonBox.AcceptRole)
        self.connect_button.clicked.connect(self.test_connection)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        saved_credentials = _load_saved_credentials()
        if saved_credentials is not None:
            self.username_input.setText(saved_credentials[0])
            self.password_input.setText(saved_credentials[1])
            self.password_input.setFocus()
        else:
            self.username_input.setFocus()

    def test_connection(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        if not username or not password:
            QMessageBox.warning(self, "Cluster login", "Enter both your username and password.")
            return
        self.connect_button.setEnabled(False)
        QApplication.processEvents()
        try:
            inspect_cluster(username, password)
        except Exception as error:
            message = str(error)
            for secret in (password, username):
                if secret:
                    message = message.replace(secret, "[redacted]")
            QMessageBox.warning(self, "Cluster login failed", message)
        else:
            self.credentials = (username, password)
            try:
                _save_credentials(username, password)
            except Exception:
                QMessageBox.warning(
                    self,
                    "Credentials not saved",
                    "Login succeeded, but Windows Credential Manager could not save the credentials.",
                )
            self.accept()
        finally:
            self.password_input.clear()
            self.connect_button.setEnabled(True)

    def take_credentials(self):
        credentials = self.credentials
        self.credentials = None
        return credentials
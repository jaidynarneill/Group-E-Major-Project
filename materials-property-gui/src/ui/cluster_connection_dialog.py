import json

from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QCheckBox,
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


def _clear_saved_credentials():
    import keyring

    saved = keyring.get_password(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT)
    if saved is not None:
        keyring.delete_password(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT)


class ClusterConnectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect to M3 cluster")
        self.setMinimumSize(560, 320)
        self.credentials = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 24)
        layout.setSpacing(22)
        title = QLabel("Cluster sign in")
        title.setStyleSheet("font-size: 24px; font-weight: 600;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(18)
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Cluster username")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Cluster password")
        for field in (self.username_input, self.password_input):
            field.setMinimumHeight(52)
            field.setStyleSheet("font-size: 20px; padding: 8px 10px;")
        form.addRow("Cluster username", self.username_input)
        form.addRow("Cluster password", self.password_input)
        layout.addLayout(form)

        self.remember_me = QCheckBox("Remember me")
        self.remember_me.setStyleSheet("font-size: 17px; spacing: 10px;")
        layout.addWidget(self.remember_me)

        buttons = QDialogButtonBox()
        self.connect_button = buttons.addButton("Connect", QDialogButtonBox.AcceptRole)
        self.connect_button.clicked.connect(self.test_connection)
        self.cancel_button = buttons.addButton("Cancel", QDialogButtonBox.RejectRole)
        buttons.rejected.connect(self.reject)
        for button in (self.connect_button, self.cancel_button):
            button.setMinimumHeight(42)
            button.setStyleSheet("font-size: 16px; padding: 6px 18px;")
        layout.addWidget(buttons)

        saved_credentials = _load_saved_credentials()
        if saved_credentials is not None:
            self.username_input.setText(saved_credentials[0])
            self.password_input.setText(saved_credentials[1])
            self.remember_me.setChecked(True)
            self.password_input.setFocus()
        else:
            self.username_input.setFocus()
        self.remember_me.toggled.connect(self._remember_me_changed)

    def _remember_me_changed(self, checked):
        if checked:
            return
        try:
            _clear_saved_credentials()
        except Exception:
            QMessageBox.warning(
                self,
                "Credentials not cleared",
                "Windows Credential Manager could not remove the saved login.",
            )

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
            if self.remember_me.isChecked():
                try:
                    _save_credentials(username, password)
                except Exception:
                    QMessageBox.warning(
                        self,
                        "Credentials not saved",
                        "Login succeeded, but Windows Credential Manager could not save the credentials.",
                    )
            else:
                try:
                    _clear_saved_credentials()
                except Exception:
                    pass
            self.accept()
        finally:
            self.password_input.clear()
            self.connect_button.setEnabled(True)

    def take_credentials(self):
        credentials = self.credentials
        self.credentials = None
        return credentials
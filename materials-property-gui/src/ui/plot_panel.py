from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QCheckBox,
    QGroupBox,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class PlotPanel(QWidget):
    MATERIALS = ["Aluminum", "Copper", "Silicon"]
    METHODS = ["MEAM", "MACE-MP", "DFT"]
    LATTICES = {
        "Aluminum": ["FCC", "BCC", "HCP"],
        "Copper": ["FCC", "BCC", "HCP"],
        "Silicon": ["Diamond cubic"],
    }

    def __init__(self):
        super().__init__()
        self.rows = []
        self.initUI()

    def initUI(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #e6e6e6;
                font-size: 14px;
            }
            QLabel#title {
                font-size: 23px;
                font-weight: bold;
                padding: 6px 0;
            }
            QLabel.header {
                color: #9cdcfe;
                font-weight: bold;
            }
            QGroupBox {
                border: 1px solid #454545;
                border-radius: 6px;
                margin-top: 12px;
                padding: 12px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QComboBox, QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 7px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                selection-background-color: #094771;
            }
            QPushButton {
                background-color: #0e639c;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QCheckBox {
                spacing: 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Materials Properties Comparison")
        title.setObjectName("title")
        layout.addWidget(title)

        config_box = QGroupBox("Simulation configurations")
        grid = QGridLayout(config_box)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)

        headers = ["Use", "Configuration", "Material", "Lattice structure", "Method"]
        for column, text in enumerate(headers):
            header = QLabel(text)
            header.setProperty("class", "header")
            header.setStyleSheet("color: #9cdcfe; font-weight: bold;")
            grid.addWidget(header, 0, column)

        for index in range(3):
            enabled = QCheckBox()
            enabled.setChecked(index == 0)

            material = QComboBox()
            material.addItems(self.MATERIALS)

            lattice = QComboBox()
            lattice.addItems(self.LATTICES[material.currentText()])

            method = QComboBox()
            method.addItems(self.METHODS)

            grid.addWidget(enabled, index + 1, 0)
            grid.addWidget(QLabel(f"Run {index + 1}"), index + 1, 1)
            grid.addWidget(material, index + 1, 2)
            grid.addWidget(lattice, index + 1, 3)
            grid.addWidget(method, index + 1, 4)

            enabled.toggled.connect(
                lambda checked, widgets=(material, lattice, method):
                    [widget.setEnabled(checked) for widget in widgets]
            )
            material.currentTextChanged.connect(
                lambda value, dropdown=lattice: self.update_lattices(dropdown, value)
            )

            if index != 0:
                material.setEnabled(False)
                lattice.setEnabled(False)
                method.setEnabled(False)

            self.rows.append((enabled, material, lattice, method))

        for column in (2, 3, 4):
            grid.setColumnStretch(column, 1)

        layout.addWidget(config_box)

        self.compare_button = QPushButton("Compare selected configurations")
        self.compare_button.clicked.connect(self.compare_runs)
        layout.addWidget(self.compare_button)

        results_box = QGroupBox("Results")
        results_layout = QVBoxLayout(results_box)

        self.status_label = QLabel(
            "Select one or more configurations, then click Compare."
        )
        self.status_label.setWordWrap(True)
        results_layout.addWidget(self.status_label)

        self.figure = Figure(figsize=(10, 6), facecolor="#1e1e1e")
        self.canvas = FigureCanvas(self.figure)
        results_layout.addWidget(self.canvas)

        layout.addWidget(results_box, stretch=1)
        self.show_empty_plots()

    def update_lattices(self, dropdown, material):
        previous_value = dropdown.currentText()
        options = self.LATTICES[material]

        dropdown.clear()
        dropdown.addItems(options)

        if previous_value in options:
            dropdown.setCurrentText(previous_value)

    def show_empty_plots(self, configurations=None):
        self.figure.clear()

        plot_titles = [
            "Equation of state",
            "Elastic constants",
            "Surface energies",
            "Derived properties",
        ]

        for index, title in enumerate(plot_titles, start=1):
            ax = self.figure.add_subplot(2, 2, index)
            ax.set_facecolor("#252526")
            ax.set_title(title, color="#e6e6e6")
            ax.tick_params(colors="#cccccc")
            for spine in ax.spines.values():
                spine.set_color("#777777")

            if configurations:
                message = "Simulation backend not connected"
            else:
                message = "Results will appear here"

            ax.text(
                0.5, 0.5, message,
                color="#bbbbbb",
                ha="center", va="center",
                transform=ax.transAxes,
                wrap=True,
            )

        self.figure.tight_layout()
        self.canvas.draw()

    def compare_runs(self):
        configurations = []

        for index, (enabled, material, lattice, method) in enumerate(self.rows, start=1):
            if enabled.isChecked():
                configurations.append(
                    f"Run {index}: {material.currentText()} / "
                    f"{lattice.currentText()} / {method.currentText()}"
                )

        if not configurations:
            self.status_label.setText("Enable at least one configuration to compare.")
            self.show_empty_plots()
            return

        self.status_label.setText(
            "Selected configurations:\n" + "\n".join(configurations)
        )
        self.show_empty_plots(configurations)
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
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

try:
    from analysis.elastic_properties import (
        EV_A3_TO_GPA,
        MEAM_CATALOG,
        birch_murnaghan,
        compute_elastic_properties,
    )
except ModuleNotFoundError as error:
    if error.name != "analysis":
        raise
    from src.analysis.elastic_properties import (
        EV_A3_TO_GPA,
        MEAM_CATALOG,
        birch_murnaghan,
        compute_elastic_properties,
    )


class CalculationWorker(QObject):
    progress = pyqtSignal(str)
    completed = pyqtSignal(object, object)
    finished = pyqtSignal()

    def __init__(self, configurations):
        super().__init__()
        self.configurations = configurations

    @pyqtSlot()
    def run(self):
        results = []
        errors = []
        try:
            for run_number, material, lattice, method in self.configurations:
                label = f"Run {run_number}: {material} / {lattice} / {method}"
                self.progress.emit(f"Calculating {label}...")
                try:
                    result = compute_elastic_properties(material, lattice, method)
                    results.append((label, result))
                except Exception as error:
                    errors.append(f"{label}: {error}")
            self.completed.emit(results, errors)
        finally:
            self.finished.emit()


class PlotPanel(QWidget):
    MATERIALS = [record["name"] for record in MEAM_CATALOG.values()]
    MATERIAL_SYMBOLS = {record["name"]: symbol for symbol, record in MEAM_CATALOG.items()}
    METHODS = ["MACE-MP", "MEAM", "DFT"]
    LATTICES = {
        record["name"]: [record["lattice_name"]]
        for record in MEAM_CATALOG.values()
    }
    DEFAULT_MATERIALS = ["Aluminum", "Copper", "Silicon"]

    def __init__(self):
        super().__init__()
        self.rows = []
        self._thread = None
        self._worker = None
        self.initUI()

    def initUI(self):
        rcParams.update({
            "font.size": 20,
            "axes.titlesize": 29,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 11,
        })
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #e6e6e6;
                font-size: 42px;
            }
            QLabel#title {
                font-size: 69px;
                font-weight: bold;
                padding: 18px 0;
            }
            QLabel.header {
                color: #9cdcfe;
                font-weight: bold;
                font-size: 42px;
            }
            QGroupBox {
                border: 1px solid #454545;
                border-radius: 6px;
                margin-top: 40px;
                padding: 36px;
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
                padding: 14px;
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
            header.setStyleSheet("color: #9cdcfe; font-weight: bold; font-size: 42px;")
            grid.addWidget(header, 0, column)

        for index in range(3):
            enabled = QCheckBox()
            enabled.setChecked(index == 0)

            material = QComboBox()
            material.addItems(self.MATERIALS)
            if index < len(self.DEFAULT_MATERIALS):
                material.setCurrentText(self.DEFAULT_MATERIALS[index])

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

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_calculations)
        layout.addWidget(self.run_button)

        results_box = QGroupBox("Results")
        results_layout = QVBoxLayout(results_box)

        self.status_label = QLabel(
            "Select configurations and click Run. Surface-energy calculations are not connected."
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

    def show_empty_plots(self, message="Run a configuration to calculate plots."):
        self.figure.clear()

        plot_titles = [
            "Equation of state",
            "Stress-strain response",
            "Elastic constants",
            "Surface energies",
        ]

        for index, title in enumerate(plot_titles, start=1):
            ax = self.figure.add_subplot(2, 2, index)
            ax.set_facecolor("#252526")
            ax.set_title(title, color="#e6e6e6")
            ax.tick_params(colors="#cccccc")
            for spine in ax.spines.values():
                spine.set_color("#777777")

            ax.text(
                0.5, 0.5, message,
                color="#bbbbbb",
                ha="center", va="center",
                transform=ax.transAxes,
                wrap=True,
            )

        self.figure.subplots_adjust(
            left=0.16, right=0.98, bottom=0.17, top=0.90, wspace=0.38, hspace=0.52
        )
        self.canvas.draw()

    def run_calculations(self):
        configurations = []

        for index, (enabled, material, lattice, method) in enumerate(self.rows, start=1):
            if enabled.isChecked():
                configurations.append((
                    index,
                    material.currentText(),
                    lattice.currentText(),
                    method.currentText(),
                ))

        if not configurations:
            self.status_label.setText("Enable at least one configuration before running.")
            return

        self.run_button.setEnabled(False)
        self.status_label.setText("Starting calculations...")

        self._thread = QThread(self)
        self._worker = CalculationWorker(configurations)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status_label.setText)
        self._worker.completed.connect(self.show_results)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._finish_run)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _finish_run(self):
        self.run_button.setEnabled(True)
        self._worker = None
        self._thread = None

    def show_results(self, results, errors):
        if not results:
            message = "No calculations completed. " + "\n".join(errors)
            self.status_label.setText(message)
            self.show_empty_plots("No calculation data available")
            return

        self._plot_eos(results)
        self._plot_stress_strain(results)
        self._plot_elastic_constants(results)
        self._plot_surface_energy_placeholder()
        summaries = []
        for label, result in results:
            summaries.append(
                f"{label}: a0={result['a0']:.4f} Å, B(EOS)={result['B_EOS']:.2f} GPa, "
                f"C11={result['C11']:.2f}, C12={result['C12']:.2f}, "
                f"C44={result['C44']:.2f}, E={result['E']:.2f} GPa, nu={result['nu']:.4f}"
            )
        summaries.append("Surface energies are not plotted: the surface-energy module is still a placeholder.")
        if errors:
            summaries.append("Failed runs: " + " | ".join(errors))
        self.status_label.setText("\n".join(summaries))
        self.figure.subplots_adjust(
            left=0.16, right=0.98, bottom=0.17, top=0.90, wspace=0.38, hspace=0.52
        )
        self.canvas.draw()

    def _new_axis(self, position, title, xlabel, ylabel):
        axis = self.figure.add_subplot(2, 2, position)
        axis.set_facecolor("#252526")
        axis.set_title(title, color="#e6e6e6")
        axis.set_xlabel(xlabel, color="#cccccc")
        axis.set_ylabel(ylabel, color="#cccccc")
        axis.tick_params(colors="#cccccc")
        axis.grid(color="#555555", alpha=0.35)
        for spine in axis.spines.values():
            spine.set_color("#777777")
        return axis

    def _plot_eos(self, results):
        self.figure.clear()
        axis = self._new_axis(1, "Equation of state", "Volume (Å³/cell)", "Energy (eV/atom)")
        for index, (label, result) in enumerate(results):
            data = result["eos_data"]
            color = f"C{index % 10}"
            axis.scatter(data["volume"], data["energy_per_atom"], color=color, label=f"{label} data")
            fit_volume = np.linspace(data["volume"].min(), data["volume"].max(), 200)
            atoms_per_cell = result["atoms_per_cell"]
            fit_energy = birch_murnaghan(
                fit_volume,
                result["E0_atom"] * atoms_per_cell,
                result["V0"],
                result["B_EOS"] / EV_A3_TO_GPA,
                result["B_prime"],
            ) / atoms_per_cell
            axis.plot(fit_volume, fit_energy, color=color, label=f"{label} fit")
        axis.legend(fontsize=14)

    def _plot_stress_strain(self, results):
        axis = self._new_axis(2, "Stress-strain response", "Strain", "Stress (GPa)")
        components = [
            ("sigma_xx_GPa", "C11", "-"),
            ("sigma_yy_GPa", "C12", "--"),
            ("sigma_xy_GPa", "C44", ":"),
        ]
        for index, (label, result) in enumerate(results):
            data = result["elastic_data"]
            strain = data["strain"]
            color = f"C{index % 10}"
            for key, component, linestyle in components:
                stress = data[key]
                fit = np.polyfit(strain, stress, 1)
                axis.scatter(strain, stress, color=color, s=12)
                axis.plot(
                    strain,
                    np.polyval(fit, strain),
                    color=color,
                    linestyle=linestyle,
                    label=f"{label} {component}",
                )
        axis.legend(fontsize=14, ncol=2)

    def _plot_elastic_constants(self, results):
        axis = self._new_axis(3, "Elastic constants", "Elastic constant", "Value (GPa)")
        names = ["C11", "C12", "C44"]
        positions = np.arange(len(names))
        width = 0.8 / len(results)
        for index, (label, result) in enumerate(results):
            offset = (index - (len(results) - 1) / 2.0) * width
            values = [result[name] for name in names]
            axis.bar(positions + offset, values, width=width, label=label)
        axis.set_xticks(positions, names)
        axis.legend(fontsize=14)

    def _plot_surface_energy_placeholder(self):
        axis = self._new_axis(4, "Surface energies", "", "")
        axis.set_axis_off()
        axis.text(
            0.5,
            0.5,
            "Surface-energy backend is not implemented.\nPlaceholder values are not plotted.",
            color="#bbbbbb",
            ha="center",
            va="center",
            transform=axis.transAxes,
            wrap=True,
        )
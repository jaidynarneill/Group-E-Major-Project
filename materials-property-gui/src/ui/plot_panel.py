from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QCheckBox,
    QGroupBox,
    QScrollArea,
    QPlainTextEdit,
    QApplication,
    QSplitter,
)
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QPalette
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

try:
    from analysis.elastic_properties import (
        EV_A3_TO_GPA,
        MEAM_CATALOG,
        MACE_MODEL_OPTIONS,
        birch_murnaghan,
        create_calculator,
        compute_elastic_properties,
        mace_model_for_method,
    )
    from analysis.surface_energy import compute_surface_energies
    from analysis.result_cache import CalculationResultCache
except ModuleNotFoundError as error:
    if error.name != "analysis":
        raise
    from src.analysis.elastic_properties import (
        EV_A3_TO_GPA,
        MEAM_CATALOG,
        MACE_MODEL_OPTIONS,
        birch_murnaghan,
        create_calculator,
        compute_elastic_properties,
        mace_model_for_method,
    )
    from src.analysis.surface_energy import compute_surface_energies
    from src.analysis.result_cache import CalculationResultCache


class CalculationWorker(QObject):
    progress = pyqtSignal(str)
    completed = pyqtSignal(object, object)
    finished = pyqtSignal()

    def __init__(self, configurations):
        super().__init__()
        self.configurations = configurations
        self.result_cache = CalculationResultCache()

    @pyqtSlot()
    def run(self):
        results = []
        errors = []
        try:
            for run_number, material, lattice, method in self.configurations:
                label = f"Run {run_number}: {material} / {lattice} / {method}"
                try:
                    cached_result = self.result_cache.load(material, lattice, method)
                except Exception as error:
                    cached_result = None
                    self.progress.emit(f"Cache check failed for {label}: {error}")
                if cached_result is not None:
                    self.progress.emit(f"Loading cached results for {label}...")
                    results.append((label, cached_result))
                    continue

                self.progress.emit(f"Calculating {label}...")
                try:
                    model = mace_model_for_method(method)
                    calculator = create_calculator(material, method, model=model)
                    result = compute_elastic_properties(
                        material,
                        lattice,
                        method,
                        model=model,
                        calculator=calculator,
                    )
                    try:
                        result["surface_energy_data"] = compute_surface_energies(
                            material,
                            lattice,
                            method,
                            calculator=calculator,
                            bulk_energy_per_atom=result["E0_atom"],
                            lattice_parameter=result["a0"],
                            model=model,
                        )
                    except Exception as error:
                        result["surface_energy_data"] = {"surfaces": []}
                        errors.append(f"{label} surface energies: {error}")
                    else:
                        try:
                            self.result_cache.save(material, lattice, method, result)
                        except Exception as error:
                            errors.append(f"{label} cache save: {error}")
                    results.append((label, result))
                except Exception as error:
                    errors.append(f"{label}: {error}")
            self.completed.emit(results, errors)
        finally:
            self.finished.emit()


class PlotPanel(QWidget):
    themeChanged = pyqtSignal(bool)

    MATERIALS = [record["name"] for record in MEAM_CATALOG.values()]
    MATERIAL_SYMBOLS = {record["name"]: symbol for symbol, record in MEAM_CATALOG.items()}
    METHODS = [*MACE_MODEL_OPTIONS, "MEAM", "DFT"]
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
        self._last_results = []
        self._last_errors = []
        self.theme_colors = {}
        self.initUI()

    def initUI(self):
        rcParams.update({
            "font.size": 19,
            "axes.titlesize": 26,
            "axes.labelsize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 10,
        })
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title_row = QHBoxLayout()
        title = QLabel("Materials Properties Comparison")
        title.setObjectName("title")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(QLabel("Theme"))
        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["Dark", "Light"])
        self.theme_selector.currentTextChanged.connect(self.apply_theme)
        title_row.addWidget(self.theme_selector)
        layout.addLayout(title_row)

        config_box = QGroupBox("Simulation configurations")
        grid = QGridLayout(config_box)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)

        headers = ["Use", "Configuration", "Material", "Lattice structure", "Method"]
        for column, text in enumerate(headers):
            header = QLabel(text)
            header.setObjectName("columnHeader")
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
            "Select configurations and click Run to calculate elastic properties and surface energies."
        )
        self.status_label.setWordWrap(True)
        results_layout.addWidget(self.status_label)

        self.result_output = QPlainTextEdit()
        self.result_output.setReadOnly(True)
        self.result_output.setPlaceholderText("Per-run calculation details will appear here.")
        self.result_output.setMinimumHeight(120)
        self.result_output.setStyleSheet("QPlainTextEdit { font-size: 18px; }")

        self.figure = Figure(figsize=(13, 23), facecolor="#1e1e1e")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(1300, 2300)
        self.output_scroll = QScrollArea()
        self.output_scroll.setWidgetResizable(True)
        self.output_scroll.setMinimumHeight(250)
        self.output_scroll.setWidget(self.canvas)

        self.output_splitter = QSplitter(Qt.Vertical)
        self.output_splitter.setHandleWidth(8)
        self.output_splitter.addWidget(self.result_output)
        self.output_splitter.addWidget(self.output_scroll)
        self.output_splitter.setStretchFactor(0, 0)
        self.output_splitter.setStretchFactor(1, 1)
        self.output_splitter.setSizes([280, 900])
        results_layout.addWidget(self.output_splitter, stretch=1)

        layout.addWidget(results_box, stretch=1)
        self.apply_theme(self.theme_selector.currentText())

    def apply_theme(self, theme_name):
        dark = theme_name == "Dark"
        colors = {
            "window": "#1e1e1e" if dark else "#f4f5f7",
            "surface": "#252526" if dark else "#ffffff",
            "control": "#2d2d2d" if dark else "#ffffff",
            "text": "#e6e6e6" if dark else "#202124",
            "secondary_text": "#cccccc" if dark else "#454a50",
            "muted_text": "#bbbbbb" if dark else "#555b63",
            "border": "#555555" if dark else "#aeb4bc",
            "grid": "#555555" if dark else "#d9dde3",
            "accent": "#0e639c" if dark else "#1769aa",
            "accent_hover": "#1177bb" if dark else "#0f568d",
            "selection": "#094771" if dark else "#cfe5fb",
        }
        self.theme_colors = colors

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(colors["window"]))
        palette.setColor(QPalette.WindowText, QColor(colors["text"]))
        palette.setColor(QPalette.Base, QColor(colors["surface"]))
        palette.setColor(QPalette.AlternateBase, QColor(colors["window"]))
        palette.setColor(QPalette.ToolTipBase, QColor(colors["surface"]))
        palette.setColor(QPalette.ToolTipText, QColor(colors["text"]))
        palette.setColor(QPalette.Text, QColor(colors["text"]))
        palette.setColor(QPalette.Button, QColor(colors["control"]))
        palette.setColor(QPalette.ButtonText, QColor(colors["text"]))
        palette.setColor(QPalette.Highlight, QColor(colors["selection"]))
        palette.setColor(QPalette.HighlightedText, QColor(colors["text"]))
        app = QApplication.instance()
        if app is not None:
            app.setPalette(palette)
            app.setStyleSheet(f"""
                QMainWindow, QDialog, QWidget {{
                    background-color: {colors['window']};
                    color: {colors['text']};
                }}
                QMenu, QComboBox QAbstractItemView {{
                    background-color: {colors['surface']};
                    color: {colors['text']};
                    selection-background-color: {colors['selection']};
                }}
                QToolTip {{
                    background-color: {colors['surface']};
                    color: {colors['text']};
                    border: 1px solid {colors['border']};
                }}
            """)

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {colors['window']};
                color: {colors['text']};
                font-size: 20px;
            }}
            QLabel#title {{ font-size: 36px; font-weight: bold; padding: 10px 0; }}
            QLabel#columnHeader {{ color: {colors['accent']}; font-weight: bold; font-size: 22px; }}
            QGroupBox {{
                border: 1px solid {colors['border']};
                border-radius: 6px;
                margin-top: 22px;
                padding: 18px;
                font-weight: bold;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}
            QComboBox, QPushButton, QPlainTextEdit {{
                background-color: {colors['control']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 9px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors['surface']};
                color: {colors['text']};
                selection-background-color: {colors['selection']};
            }}
            QPushButton {{ background-color: {colors['accent']}; font-weight: bold; }}
            QPushButton:hover {{ background-color: {colors['accent_hover']}; }}
            QCheckBox {{ spacing: 8px; }}
        """)
        self.figure.set_facecolor(colors["surface"])
        if self._last_results:
            self.show_results(self._last_results, self._last_errors)
        else:
            self.show_empty_plots()
        self.themeChanged.emit(dark)

    def update_lattices(self, dropdown, material):
        previous_value = dropdown.currentText()
        options = self.LATTICES[material]

        dropdown.clear()
        dropdown.addItems(options)

        if previous_value in options:
            dropdown.setCurrentText(previous_value)

    def show_empty_plots(self, message="Run a configuration to calculate plots."):
        colors = self.theme_colors
        self.figure.clear()
        self.figure.set_facecolor(colors["surface"])

        plot_titles = [
            "Equation Of State",
            "Stress-Strain Response",
            "Elastic Constants",
            "Surface Energies",
        ]

        for index, title in enumerate(plot_titles, start=1):
            ax = self.figure.add_subplot(2, 2, index)
            ax.set_facecolor(colors["surface"])
            ax.set_title(title, color=colors["text"])
            ax.tick_params(colors=colors["secondary_text"])
            for spine in ax.spines.values():
                spine.set_color(colors["border"])

            ax.text(
                0.5, 0.5, message,
                color=colors["muted_text"],
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
        self.result_output.clear()

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
        self._last_results = results
        self._last_errors = errors
        if not results:
            message = "No calculations completed. " + "\n".join(errors)
            self.status_label.setText("Run finished with errors.")
            self.result_output.setPlainText(message)
            self.show_empty_plots("No calculation data available")
            return

        self._plot_eos(results)
        self._plot_stress_strain(results)
        self._plot_elastic_constants(results)
        self._plot_surface_energies(results)
        summaries = []
        for label, result in results:
            run_name = label.split(":", 1)[0]
            c11 = result["C11"]
            c12 = result["C12"]
            c44 = result["C44"]
            eos_bulk_modulus = result["B_EOS"]
            elastic_bulk_modulus = result["B_Cij"]
            bulk_difference = elastic_bulk_modulus - eos_bulk_modulus
            bulk_difference_percent = (
                abs(bulk_difference) / abs(eos_bulk_modulus) * 100.0
                if eos_bulk_modulus != 0.0
                else float("inf")
            )
            stiffness_matrix = np.array([
                [c11, c12, c12, 0.0, 0.0, 0.0],
                [c12, c11, c12, 0.0, 0.0, 0.0],
                [c12, c12, c11, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, c44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, c44, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, c44],
            ])
            summaries.extend([
                f"{run_name}: {result['material']} | {result['lattice_structure']} | {result['method']}",
                "  EOS:",
                f"    Equilibrium volume = {result['V0']:.4f} Angstrom^3",
                f"    Equilibrium energy = {result['E0_atom'] * result['atoms_per_cell']:.6f} eV",
                f"    Bulk modulus       = {eos_bulk_modulus:.3f} GPa",
                f"    B'                 = {result['B_prime']:.3f}",
                "  Elastic constants:",
                f"    C11 = {c11:.3f} GPa",
                f"    C12 = {c12:.3f} GPa",
                f"    C44 = {c44:.3f} GPa",
                "  Linear fit R^2:",
                f"    C11 fit = {result['R2_C11']:.8f}",
                f"    C12 fit = {result['R2_C12']:.8f}",
                f"    C44 fit = {result['R2_C44']:.8f}",
                "  Bulk modulus:",
                f"    From elastic constants = {elastic_bulk_modulus:.3f} GPa",
                f"    From EOS               = {eos_bulk_modulus:.3f} GPa",
                f"    Difference             = {bulk_difference:.3f} GPa",
                f"    Difference             = {bulk_difference_percent:.2f}%",
                "  Polycrystalline properties:",
                f"    Voigt shear modulus = {result['G_V']:.3f} GPa",
                f"    Reuss shear modulus = {result['G_R']:.3f} GPa",
                f"    Hill shear modulus  = {result['G_H']:.3f} GPa",
                f"    Young's modulus     = {result['E']:.3f} GPa",
                f"    Poisson's ratio     = {result['nu']:.4f}",
                "  Stiffness matrix [GPa]:",
            ])
            summaries.extend(
                "    " + " ".join(f"{value:.3f}" for value in row)
                for row in stiffness_matrix
            )
            summaries.append("  Surface energies:")
            surfaces = result.get("surface_energy_data", {}).get("surfaces", [])
            if surfaces:
                for surface_result in surfaces:
                    converged = "converged" if surface_result["converged"] else "not converged"
                    summaries.append(
                        f"    {surface_result['orientation']} = {surface_result['energy_eV_A2']:.6f} eV/Å² "
                        f"({converged}; {surface_result['atom_count']} atoms; {surface_result['area_A2']:.2f} Å²)"
                    )
            else:
                summaries.append("    No surface-energy data returned.")
            summaries.append("")
        if errors:
            summaries.extend(["Run Errors", *[f"  {error}" for error in errors]])
        self.status_label.setText("Run completed." if not errors else "Run completed with errors.")
        report = "\n".join(summaries)
        self.result_output.setPlainText(report)
        print(report)
        self.figure.subplots_adjust(
            left=0.13, right=0.98, bottom=0.12, top=0.96, wspace=0.32, hspace=0.40
        )
        self.canvas.draw()

    def _new_axis(self, position, title, xlabel, ylabel):
        colors = self.theme_colors
        axis = self.figure.add_subplot(2, 2, position)
        axis.set_facecolor(colors["surface"])
        axis.set_title(title, color=colors["text"])
        axis.set_xlabel(xlabel, color=colors["secondary_text"])
        axis.set_ylabel(ylabel, color=colors["secondary_text"])
        axis.tick_params(colors=colors["secondary_text"])
        axis.grid(color=colors["grid"], alpha=0.65 if self.theme_selector.currentText() == "Light" else 0.35)
        for spine in axis.spines.values():
            spine.set_color(colors["border"])
        return axis

    def _plot_eos(self, results):
        self.figure.clear()
        axis = self._new_axis(1, "Equation Of State", "Volume (Å³/cell)", "Energy (eV/atom)")
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
        axis = self._new_axis(2, "Stress-Strain Response", "Strain", "Stress (GPa)")
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
        axis = self._new_axis(3, "Elastic Constants", "Elastic Constant", "Value (GPa)")
        axis.grid(False)
        names = ["C11", "C12", "C44", "Bulk Modulus"]
        positions = np.arange(len(names))
        width = 0.8 / len(results)
        for index, (label, result) in enumerate(results):
            offset = (index - (len(results) - 1) / 2.0) * width
            values = [result["C11"], result["C12"], result["C44"], result["B_Cij"]]
            axis.bar(positions + offset, values, width=width, label=label)
        axis.set_xticks(positions, names)
        axis.legend(fontsize=14)

    def _plot_surface_energies(self, results):
        axis = self._new_axis(4, "Surface Energies", "Surface Orientation", "Energy (eV/Å²)")
        axis.grid(False)
        available = [
            (label, result, result.get("surface_energy_data", {}).get("surfaces", []))
            for label, result in results
        ]
        available = [entry for entry in available if entry[2]]
        if not available:
            axis.text(
                0.5,
                0.5,
                "No surface-energy results available",
                color=self.theme_colors["muted_text"],
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            return

        orientations = sorted({
            surface_result["orientation"]
            for _, _, surfaces in available
            for surface_result in surfaces
        })
        positions = np.arange(len(orientations))
        width = 0.8 / len(available)
        for index, (label, _, surfaces) in enumerate(available):
            energy_by_orientation = {
                surface_result["orientation"]: surface_result["energy_eV_A2"]
                for surface_result in surfaces
            }
            values = [energy_by_orientation.get(orientation, np.nan) for orientation in orientations]
            offset = (index - (len(available) - 1) / 2.0) * width
            axis.bar(positions + offset, values, width=width, label=label)
        axis.set_xticks(positions, orientations)
        axis.legend(fontsize=10)
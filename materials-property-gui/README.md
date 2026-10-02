# Materials Property GUI

This project is a graphical user interface (GUI) designed to predict surface energies and elastic properties of materials, specifically copper, aluminum, and silicon, using many-body potentials such as MEAM and S-W. The GUI allows users to select materials, lattice structures, and computational methods, and visualize the results through plots.

## Project Structure

```
materials-property-gui
├── src
│   ├── main.py                # Entry point for the GUI application
│   ├── ui
│   │   ├── main_window.py     # Defines the main window layout and integration
│   │   ├── selection_panel.py  # Implements the selection panel with dropdowns
│   │   └── plot_panel.py       # Displays output plots for surface energies and elastic constants
│   ├── analysis
│   │   ├── surface_energy.py   # Functions to calculate surface energies
│   │   └── elastic_properties.py # Functions to compute elastic properties
│   └── data
│       └── README.md           # Documentation for data formats and sources
├── tests
│   ├── test_surface_energy.py   # Unit tests for surface energy calculations
│   └── test_elastic_properties.py # Unit tests for elastic properties calculations
├── requirements.txt             # Lists project dependencies
└── README.md                    # Project documentation and usage guidelines
```

## Features

- **Material Selection**: Choose from silicon, copper, or aluminum.
- **Lattice Structure Selection**: Options for FCC and other structures.
- **Method Selection**: Select computational methods including MEAM, MACE-MP, and DFT.
- **Comparison Rows**: Compare results across different configurations.
- **Visualization**: Display plots for surface energies and elastic constants based on selected parameters.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd materials-property-gui
pip install -r requirements.txt
```

## Usage

Run the application using the following command:

```bash
python src/main.py
```
Follow the on-screen instructions to select materials, structures, and methods, and view the resulting plots.

## Elastic Calculation Backends

The elastic-properties module performs EOS and small-strain stress calculations for cubic FCC, BCC, diamond, and simple-cubic structures. The material list is read from the elemental records in `src/data/potentials/library.meam`; non-cubic records such as HCP magnesium are excluded because the current backend fits cubic elastic constants. MACE-MP requires `ase` and `mace-torch`. MEAM uses ASE's LAMMPS calculator; install LAMMPS separately. The bundled `library.meam` contains single-element records, so no separate MEAM parameter file is needed for these elemental calculations. The app selects a canonical element label from the library; set `MEAM_LIBRARY` to use another library file or `LAMMPS_COMMAND` if the executable is not named `lmp`. `MEAM_PAIR_COEFF` can override the generated LAMMPS `pair_coeff` line and accepts `{library}`, `{parameter}`, and `{element}` placeholders.

Surface energies are calculated from relaxed ASE slabs for the low-index (100), (110), and (111) orientations. The reported value is `gamma = (E_slab - N * E_bulk) / (2 * A)` in eV/Å², where `A` is one exposed face area; the factor of two accounts for both slab faces. Slabs default to 8 layers, 10 Å vacuum, and a 0.05 eV/Å force tolerance. These are starting settings and should be checked for slab-thickness and relaxation convergence before reporting results.

DFT uses Quantum ESPRESSO through ASE. Install `pw.x`, place the selected element's `.UPF` file under `src/data/pseudopotentials/` or set `ESPRESSO_PSEUDO_DIR`, and select the file with `DFT_PSEUDO_AL`, `DFT_PSEUDO_CU`, or `DFT_PSEUDO_SI`. Set `ESPRESSO_COMMAND` if needed. Plane-wave cutoffs default to 60/480 Ry and can be overridden with `DFT_ECUTWFC_RY` and `DFT_ECUTRHO_RY`; converge these settings for the chosen pseudopotential before using results.

DFT pseudopotentials, the LAMMPS executable, and the Quantum ESPRESSO executable are not bundled with this repository. The calculation function reports a setup error until the requested backend is configured. For example:

```python
from src.analysis.elastic_properties import compute_elastic_properties

results = compute_elastic_properties("Aluminum", "FCC", "MACE-MP")
print(results["C11"], results["C12"], results["C44"])
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
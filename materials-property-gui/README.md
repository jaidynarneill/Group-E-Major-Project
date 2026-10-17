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

## Build A Windows Executable

Install the app and build dependencies in the same Python environment, then run the build script from PowerShell:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-build.txt
.\build_windows.ps1
```

The executable is `dist/MaterialsPropertyGUI/MaterialsPropertyGUI.exe`; keep the `dist/MaterialsPropertyGUI` folder together when moving or sharing the build. This bundles the GUI, Python packages, `src/data`, and the Slurm launcher. LAMMPS and the VASP cluster are external. MACE downloads its selected checkpoint the first time it runs, so that first calculation needs internet access.

## Elastic Calculation Backends

The elastic-properties module performs EOS and small-strain stress calculations for cubic FCC, BCC, diamond, and simple-cubic structures. The material list is read from the elemental records in `src/data/potentials/library.meam`; non-cubic records such as HCP magnesium are excluded because the current backend fits cubic elastic constants. MACE-MP requires `ase` and `mace-torch`. The method dropdown provides MACE-MP Small, Medium, and Large, plus MACE-MPA (the `medium-mpa-0` checkpoint). MEAM uses ASE's LAMMPS calculator; install LAMMPS separately. The bundled `library.meam` contains single-element records, so no separate MEAM parameter file is needed for these elemental calculations. The app selects a canonical element label from the library; set `MEAM_LIBRARY` to use another library file or `LAMMPS_COMMAND` if the executable is not named `lmp`. `MEAM_PAIR_COEFF` can override the generated LAMMPS `pair_coeff` line and accepts `{library}`, `{parameter}`, and `{element}` placeholders.

Surface energies are calculated from relaxed ASE slabs for the low-index (100), (110), and (111) orientations. The reported value is `gamma = (E_slab - N * E_bulk) / (2 * A)` in eV/Å², where `A` is one exposed face area; the factor of two accounts for both slab faces. Slabs default to 8 layers, 10 Å vacuum, and a 0.05 eV/Å force tolerance. These are starting settings and should be checked for slab-thickness and relaxation convergence before reporting results.

DFT runs remotely through VASP 6.6.1 on `m3.massive.org.au`. The app uses `/projects/lh36/sdwi0002/opt/modulefiles` and loads only `vasp/6.6.1`; the VASP module supplies its runtime dependencies. Jobs request Intel nodes as required by this private VASP build and use 16 MPI tasks. Surface slab jobs request 32 GB of memory; bulk, EOS, and elastic jobs retain the cluster's default memory request. The cluster must provide `vasp_std`, `potgen`, and Slurm. Each Slurm phase runs `potgen <element>` (for example, `potgen Al`) to generate `POTCAR` on the cluster. The workflow first relaxes the bulk cell, then centers 13 static EOS calculations on its relaxed volume, runs 12 fixed-cell elastic strains, and relaxes three low-index slabs. Gamma meshes use a total grid density of 1464 divided by the number of atoms, scaled for reciprocal-cell lengths; cobalt calculations start spin-polarized with 0.6 μB per atom. `ENCUT` is 1.3 times the generated POTCAR's `ENMAX`. Each phase is limited to two hours and returns the same report and plots used by local methods. Check convergence and magnetic settings for each material before treating values as publication-quality.

Inputs, generated POTCAR files, VASP outputs, and Slurm logs are retained directly under `/fs04/scratch2/lh36/jarn0012/vasp/<Material>-Dft`, in phase folders such as `bulk-relax`, `eos`, `elastic`, and `surfaces` (for example, `Aluminum-Dft/eos`). Before submitting jobs, the app checks the shared material folder (case-insensitively) and inspects saved outputs. It reuses a phase only when its retained inputs match the selected material, lattice, and current calculation settings, and its outputs are readable. A complete set of phases is reused as-is; a run complete through EOS and elasticity resumes only its missing surface tasks; earlier, incomplete, or incompatible stages trigger fresh phase folders directly under the material folder. Nested `run-*` folders are not searched or created. If a phase folder already exists, a retry uses a uniquely named sibling phase folder rather than overwriting the previous attempt. The app downloads calculation outputs and assembles the report locally. For fresh runs it removes the obsolete shared `/fs04/scratch2/lh36/jarn0012/vasp/slurm` directory. POTCAR contents are never downloaded to the laptop or stored in the cache. The SSH server must already be trusted in the user's SSH known-hosts file.

Completed calculation results, including the data used for plots and the printed report, are cached by material, lattice, method, and model/backend. Source runs store the cache under `%LOCALAPPDATA%\MaterialsPropertyGUI\calculations` on Windows, or `$XDG_CACHE_HOME/MaterialsPropertyGUI/calculations` on other platforms (`~/.cache` is used when `XDG_CACHE_HOME` is unset). Packaged Windows builds store it in a `calculations` folder beside the executable, so moving the complete app folder also moves its cache. Set `MATERIALS_PROPERTY_CACHE` to use another directory. DFT cache entries identify the VASP backend and workflow settings; remote job files are not cached. Cluster login credentials are stored separately in Windows Credential Manager, not in the app cache.

For example, the shared local analysis API remains available for non-DFT calculators:

```python
from src.analysis.elastic_properties import compute_elastic_properties

results = compute_elastic_properties("Aluminum", "FCC", "MACE-MP")
print(results["C11"], results["C12"], results["C44"])
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
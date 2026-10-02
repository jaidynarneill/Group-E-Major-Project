import numpy as np
import matplotlib.pyplot as plt

def calculate_surface_energy(material, lattice_structure, method):
    """
    Calculate the surface energy for a given material, lattice structure, and method.
    
    Parameters:
    - material: str, material type ('silicon', 'copper', 'aluminum')
    - lattice_structure: str, lattice structure type ('FCC', 'BCC', etc.)
    - method: str, method used for calculation ('MEAM', 'MACE-MP', 'DFT')
    
    Returns:
    - surface_energy: float, calculated surface energy in eV/Å²
    """
    # Placeholder for actual calculation logic
    if material == 'silicon':
        if lattice_structure == 'FCC':
            if method == 'MEAM':
                surface_energy = 1.5  # Example value
            elif method == 'MACE-MP':
                surface_energy = 1.6  # Example value
            elif method == 'DFT':
                surface_energy = 1.4  # Example value
        else:
            surface_energy = 1.2  # Example value for other structures
    elif material == 'copper':
        surface_energy = 1.3  # Example value
    elif material == 'aluminum':
        surface_energy = 1.1  # Example value
    else:
        raise ValueError("Material not recognized.")
    
    return surface_energy

def plot_surface_energy(material, lattice_structure, method):
    """
    Plot the surface energy for the selected parameters.
    
    Parameters:
    - material: str, material type
    - lattice_structure: str, lattice structure type
    - method: str, method used for calculation
    """
    surface_energy = calculate_surface_energy(material, lattice_structure, method)
    
    plt.figure()
    plt.bar(f"{material} ({lattice_structure})", surface_energy)
    plt.ylabel("Surface Energy (eV/Å²)")
    plt.title(f"Surface Energy Calculation using {method}")
    plt.grid()
    plt.tight_layout()
    plt.show()
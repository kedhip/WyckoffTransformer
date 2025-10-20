"""
Visualize the crystal structure generation process from Wyckoff Transformer tokens
and DiffCSP++ diffusion steps.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import pickle
import gzip
from typing import List, Tuple, Optional
import torch
from pymatgen.core import Structure
from pymatgen.vis.structure_vtk import StructureVis
from mpl_toolkits.mplot3d import Axes3D

def load_cached_data(dataset: str = "mp_20") -> dict:
    """Load cached dataset containing Wyckoff positions and structures."""
    cache_folder = Path(__file__).parent.parent / "cache"
    cache_file = cache_folder / dataset / "data.pkl.gz"
    with gzip.open(cache_file, "rb") as f:
        data = pickle.load(f)
    return data

def create_structure_sequence(
    lattice: np.ndarray,
    frac_coords_sequence: List[np.ndarray],
    atomic_numbers: List[int]
) -> List[Structure]:
    """Create a sequence of pymatgen Structure objects from diffusion steps."""
    structures = []
    for coords in frac_coords_sequence:
        struct = Structure(
            lattice=lattice,
            species=atomic_numbers,
            coords=coords,
            coords_are_cartesian=False
        )
        structures.append(struct)
    return structures

def animate_diffusion_3d(
    frac_coords_sequence: List[np.ndarray],
    lattice: np.ndarray,
    atomic_numbers: List[int],
    save_path: Optional[str] = None,
    interval: int = 200
) -> None:
    """
    Create an animation of the diffusion process in 3D.
    
    Args:
        frac_coords_sequence: List of fractional coordinates at each step
        lattice: Crystal lattice matrix
        atomic_numbers: List of atomic numbers for each site
        save_path: Optional path to save animation
        interval: Time interval between frames in milliseconds
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up plot limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    
    scatter = ax.scatter([], [], [], c='b', marker='o')
    
    def init():
        scatter._offsets3d = ([], [], [])
        return scatter,
    
    def update(frame):
        coords = frac_coords_sequence[frame]
        scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
        ax.set_title(f'Diffusion Step {frame}')
        return scatter,
    
    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=len(frac_coords_sequence),
        interval=interval, blit=True
    )
    
    if save_path:
        anim.save(save_path, writer='pillow')
    
    plt.show()

def visualize_final_structure(structure: Structure) -> None:
    """Visualize the final structure using pymatgen's structure visualizer."""
    vis = StructureVis()
    vis.set_structure(structure)
    vis.show()

def example_visualization():
    """Example usage of the visualization tools."""
    # Load example data (you would replace this with your actual data)
    # Example: Create a simple cubic lattice
    lattice = np.eye(3) * 5.0
    
    # Create example diffusion sequence (random walk)
    n_sites = 4
    n_steps = 30
    start_coords = np.random.rand(n_sites, 3)
    noise_scale = 0.01
    
    frac_coords_sequence = []
    current_coords = start_coords.copy()
    
    for _ in range(n_steps):
        noise = np.random.normal(0, noise_scale, current_coords.shape)
        current_coords = (current_coords + noise) % 1.0  # Keep in unit cell
        frac_coords_sequence.append(current_coords.copy())
    
    # Example atomic numbers (all carbon for simplicity)
    atomic_numbers = [6] * n_sites
    
    # Create animation
    animate_diffusion_3d(
        frac_coords_sequence,
        lattice,
        atomic_numbers,
        save_path="diffusion_animation.gif"
    )
    
    # Create final structure
    final_structure = Structure(
        lattice=lattice,
        species=atomic_numbers,
        coords=frac_coords_sequence[-1],
        coords_are_cartesian=False
    )
    
    # Visualize final structure
    visualize_final_structure(final_structure)

if __name__ == "__main__":
    example_visualization()

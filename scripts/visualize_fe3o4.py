"""
Fe3O4 (Magnetite) Crystal Structure Visualization

This script visualizes the crystal structure of Fe3O4 with correct Wyckoff positions
in space group 227 (Fd-3m). It can use either:

1. WyFormer-generated structure data from Figshare:
   - Dataset: https://figshare.com/articles/dataset/WyFormer_generated_structures/29094701
   - Download the dataset and provide the path to:
     mp_20/WyckoffTransformer/DiffCSP++10k/data.csv.gz
     
2. CIF file input:
   - Provide a CIF file containing Fe3O4 structure
   
3. Or generate a random structure with correct Wyckoff positions:
   - Fe3+ at tetrahedral 8a sites (1/8, 1/8, 1/8)
   - Fe2+/Fe3+ at octahedral 16d sites (1/2, 1/2, 1/2)
   - O2- at 32e sites (x, x, x) with x ≈ 0.255

Example usage:
    # Using WyFormer data:
    python visualize_fe3o4.py --wyformer path/to/data.csv.gz
    
    # Using CIF file:
    python visualize_fe3o4.py --cif path/to/fe3o4.cif
    
    # Using random generation:
    python visualize_fe3o4.py
    
    # Save animation:
    python visualize_fe3o4.py --save animation.gif
"""

import numpy as np
import pandas as pd
import json
import argparse
import logging
from pathlib import Path
from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.cif import CifParser
from pyxtal import pyxtal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def validate_fe3o4_structure(structure):
    """
    Validate that a structure has the correct Fe3O4 composition and symmetry.
    
    Args:
        structure: pymatgen Structure object
    Returns:
        bool: True if valid Fe3O4 structure, False otherwise
    """
    try:
        # Check composition
        composition = structure.composition
        fe_atoms = composition['Fe']
        o_atoms = composition['O']
        
        if not (23.5 <= fe_atoms <= 24.5 and 31.5 <= o_atoms <= 32.5):
            logging.warning(f"Invalid composition: Fe={fe_atoms}, O={o_atoms}")
            return False
            
        # Check lattice parameters (should be cubic)
        a, b, c = structure.lattice.abc
        angles = structure.lattice.angles
        if not (abs(a - b) < 0.1 and abs(b - c) < 0.1 and 
                all(abs(angle - 90) < 0.1 for angle in angles)):
            logging.warning(f"Non-cubic lattice: a={a}, b={b}, c={c}, angles={angles}")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Error validating structure: {str(e)}")
        return False

def load_structure_from_cif(cif_path):
    """
    Load a structure from a CIF file
    """
    parser = CifParser(cif_path)
    structure = parser.get_structures()[0]
    return structure

def load_wyformer_data(wyformer_path):
    """
    Load WyFormer dataset.
    Args:
        wyformer_path: Path to the WyFormer dataset file (.csv.gz)
    Returns:
        DataFrame containing WyFormer structures
    """
    df = pd.read_csv(wyformer_path, index_col="material_id")
    return df

def filter_fe3o4_structures(df):
    """
    Filter WyFormer data for Fe3O4 structures with correct space group and Wyckoff positions.
    Args:
        df: DataFrame containing WyFormer structures
    Returns:
        List of suitable Fe3O4 structures
    """
    # Filter for space group 227 (Fd-3m)
    sg_227 = df[df['spacegroup_number'] == 227]
    
    # Filter for Fe3O4 composition
    fe3o4_structures = []
    for idx, row in sg_227.iterrows():
        if not isinstance(row['elements'], list):
            row['elements'] = eval(row['elements'])
        if not isinstance(row['wyckoff_letters'], list):
            row['wyckoff_letters'] = eval(row['wyckoff_letters'])
        if not isinstance(row['multiplicity'], list):
            row['multiplicity'] = eval(row['multiplicity'])
            
        # Check elements and multiplicities
        if (set(row['elements']) == {'Fe', 'O'} and
            # Should have 8 Fe in 8a, 16 Fe in 16d, and 32 O in 32e sites
            row['wyckoff_letters'] == ['a', 'd', 'e'] and
            row['multiplicity'] == [8, 16, 32]):
            fe3o4_structures.append(row)
    
    return fe3o4_structures

def create_fe3o4_structure(wyformer_path=None, cif_path=None):
    """
    Create Fe3O4 structure with correct Wyckoff positions using:
    1. WyFormer data if wyformer_path provided
    2. CIF file if cif_path provided
    3. Random generation if neither provided
    
    Args:
        wyformer_path: Optional path to WyFormer dataset file
        cif_path: Optional path to CIF file
    
    Returns:
        pymatgen Structure object with Fe3O4 crystal structure
    """
    # Try loading from CIF if provided
    if cif_path and Path(cif_path).exists():
        try:
            return load_structure_from_cif(cif_path)
        except Exception as e:
            logging.error(f"Failed to load CIF file: {str(e)}")
    
    # Try loading from WyFormer data if provided
    if wyformer_path and Path(wyformer_path).exists():
        try:
            # Try to load and use WyFormer data
            df = load_wyformer_data(wyformer_path)
            structures = filter_fe3o4_structures(df)
            
            if structures:
                # Use the first suitable structure found
                structure_data = structures[0]
                if isinstance(structure_data['structure'], str):
                    structure_dict = json.loads(structure_data['structure'])
                else:
                    structure_dict = structure_data['structure']
                return Structure.from_dict(structure_dict)
                
        except Exception as e:
            logging.error(f"Failed to load WyFormer data: {str(e)}")
            logging.info("Falling back to random generation")
    
    # Fall back to random generation
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            crystal = pyxtal()
            # Set up Wyckoff positions for Fe3O4
            species = ['Fe', 'Fe', 'O']         # Fe3+, Fe2+/Fe3+ mixed, O2-
            numIons = [8, 16, 32]              # multiplicities
            sites = [['8a'], ['16d'], ['32e']]  # Wyckoff positions
            
            crystal.from_random(3, 227, species, numIons, sites, conventional_cell=True)
            logging.info(f"Successfully generated structure on attempt {attempt + 1}")
            return crystal.to_pymatgen()
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_attempts - 1:
                logging.error("Failed to generate structure after maximum attempts")
                return None
            continue

def create_diffusion_steps(structure, num_steps=30):
    """Create artificial diffusion steps by random perturbation."""
    all_structures = [structure.copy()]
    coords = structure.frac_coords.copy()
    
    for i in range(num_steps):
        # Add small random perturbations to atomic positions
        noise = np.random.normal(0, 0.01, coords.shape)
        new_coords = coords + (i/num_steps) * noise
        new_coords = new_coords % 1.0  # Wrap back to unit cell
        
        new_structure = Structure(structure.lattice, 
                                structure.species,
                                new_coords)
        all_structures.append(new_structure)
    
    return all_structures

def plot_structure_3d(structure, ax=None, show=True):
    """Plot a single structure in 3D."""
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot unit cell
    matrix = structure.lattice.matrix
    for i in range(8):
        corner = np.array([i & 1, (i >> 1) & 1, (i >> 2) & 1])
        pos = np.dot(corner, matrix)
        ax.scatter(*pos, color='gray', alpha=0.5, s=1)
    
    # Plot atoms
    coords = structure.cart_coords
    for i, (site, coord) in enumerate(zip(structure.species, coords)):
        if str(site) == 'Fe':
            # Fe ions in two different Wyckoff positions
            color = 'blue' if i < 8 else 'red'  
            size = 100
        else:
            # Oxygen ions
            color = 'green'
            size = 50
        ax.scatter(*coord, c=color, s=size)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Fe3O4 Crystal Structure\nBlue: Fe3+ (8a), Red: Fe2+/Fe3+ (16d), Green: O2- (32e)')
    
    if show:
        plt.show()
    else:
        return ax

def animate_diffusion(structures):
    """Create animation of the diffusion process."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        plot_structure_3d(structures[frame], ax=ax, show=False)
        ax.set_title(f'Diffusion Step {frame}')
        return ax,
    
    anim = animation.FuncAnimation(fig, update, 
                                 frames=len(structures), 
                                 interval=200, 
                                 blit=False)
    plt.show()
    return anim

def main():
    parser = argparse.ArgumentParser(description='Visualize Fe3O4 crystal structure')
    parser.add_argument('--wyformer', type=str, help='Path to WyFormer dataset file')
    parser.add_argument('--cif', type=str, help='Path to CIF file')
    parser.add_argument('--save', type=str, help='Path to save animation')
    args = parser.parse_args()
    
    # Create Fe3O4 structure
    logging.info("Generating Fe3O4 structure...")
    struct = create_fe3o4_structure(wyformer_path=args.wyformer, cif_path=args.cif)
    if struct is None:
        logging.error("Failed to create Fe3O4 structure. Exiting.")
        return
    
    # Validate the generated structure
    if not validate_fe3o4_structure(struct):
        logging.error("Generated structure is not a valid Fe3O4 structure. Exiting.")
        return

    # Plot initial structure
    logging.info("Plotting initial structure...")
    plot_structure_3d(struct)
    
    # Generate diffusion steps
    logging.info("Generating diffusion steps...")
    diffusion_structures = create_diffusion_steps(struct)
    
    # Animate diffusion
    logging.info("Animating diffusion process...")
    anim = animate_diffusion(diffusion_structures)
    
    # Save animation if requested
    if args.save:
        logging.info(f"Saving animation to {args.save}...")
        if args.save.endswith('.gif'):
            anim.save(args.save, writer='pillow')
        else:
            anim.save(args.save)
    
    logging.info("Done!")

if __name__ == "__main__":
    main()

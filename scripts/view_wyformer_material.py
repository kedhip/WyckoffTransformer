"""
Script to load and display a single material from WyFormer-generated structures.
"""

import pandas as pd
from pathlib import Path
import gzip
from typing import Dict, Any
from monty.json import MontyDecoder

def load_wyformer_material(material_index: int = 0) -> Dict[str, Any]:
    """Load a single material from WyFormer MP-20 generated data.
    
    Args:
        material_index (int): Index of the material to load (default: 0)
    
    Returns:
        Dict[str, Any]: Dictionary containing material properties
        
    Raises:
        FileNotFoundError: If the dataset file is not found
        IndexError: If the material index is out of range
    """
    # Path to the WyFormer generated data
    data_path = Path(__file__).parent.parent / "cdvae" / "data" / "mp_20" / "WyckoffTransformer" / "data.csv.gz"
    
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        print("Please download the dataset from Figshare: https://figshare.com/articles/dataset/WyFormer_generated_structures/29094701")
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    # Read the data using pandas (it's in CSV format with gzip compression)
    df = pd.read_csv(data_path, index_col="material_id", dtype=str).map(MontyDecoder().decode)
    
    if material_index >= len(df):
        raise IndexError(f"Material index {material_index} out of range. Total materials: {len(df)}")
    
    # Get the specified material
    material = df.iloc[material_index]
    
    # Create a formatted dict with the requested attributes
    material_info = {
        "Space Group": material["spacegroup_number"],
        "Site Symmetries": material["site_symmetries"],
        "Elements": material["elements"],
        "Multiplicities": material["multiplicity"],
        "Degrees of Freedom": material["dof"],
        "Wyckoff Letters": material["wyckoff_letters"]
    }
    
    return material_info

def display_material_info(material: Dict[str, Any]) -> None:
    """Print the material information in a readable format.
    
    Args:
        material (Dict[str, Any]): Dictionary containing material properties
    """
    try:
        if not material:
            print("No material data to display")
            return
            
        print("\nWyFormer Generated Material Information:")
        print("=" * 50)
        
        # Display space group information
        print(f"Space Group Number: {material['Space Group']}")
        
        # Display Wyckoff position details
        print("\nWyckoff Positions:")
        print("-" * 30)
        for i in range(len(material['Site Symmetries'])):
            print(f"\nPosition {i+1}:")
            print(f"  Element:           {material['Elements'][i]}")
            print(f"  Site Symmetry:     {material['Site Symmetries'][i]}")
            print(f"  Multiplicity:      {material['Multiplicities'][i]}")
            print(f"  Wyckoff Letter:    {material['Wyckoff Letters'][i]}")
            print(f"  Degrees of Freedom: {material['Degrees of Freedom'][i]}")
    except Exception as e:
        print(f"Error displaying material information: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Display information about a material from the WyFormer MP-20 dataset"
    )
    parser.add_argument(
        "--index", 
        type=int, 
        default=0,
        help="Index of the material to display (default: 0)"
    )
    args = parser.parse_args()
    
    try:
        material = load_wyformer_material(args.index)
        display_material_info(material)
    except Exception as e:
        print(f"\nError: {str(e)}")

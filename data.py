"""
This module provides functions for reading the structures and extracting the symmetry information.
"""
from itertools import repeat
from functools import partial
from pathlib import Path
import pickle
import warnings
import re
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from sklearn.model_selection import train_test_split

from preprocess_wychoffs import get_augmentation_dict


def read_cif(cif: str) -> Structure:
    """
    Reads a CIF file and returns the structure.

    Args:
        cif (str): The CIF file content.

    Returns:
        Structure: The structure.
    """
    return CifParser.from_str(cif).parse_structures(primitive=False)[0]


def structure_to_sites(
    structure: Structure,
    wychoffs_enumerated_by_ss: dict,
    wychoffs_augmentation: dict = None,
    tol: float = 0.1) -> dict:
    """
    Converts a pymatgen structure to a dictionary of symmetry sites.

    Args:
        structure (Structure): The pymatgen structure.
        wychoffs_enumerated_by_ss (dict)
        wychoffs_augmentation (dict, optional)
        tol (float, optional): The tolerance passed to pyxtal().from_seed.
            Defaults to 0.1, as in Materials Project.

    Returns:
        dict: A dictionary with the following keys:
            - symmetry_sites: list of lists of strings, where each string is a site symmetry
            - symmetry_elements: list of pymatgen Element objects
            - spacegroup_number: int
    """
    pyxtal_structure = pyxtal()
    pyxtal_structure.from_seed(structure, tol=tol)

    elements = [Element(site.specie) for site in pyxtal_structure.atom_sites]
    electronegativity = [element.X for element in elements]
    wyckoffs = [site.wp for site in pyxtal_structure.atom_sites]
    for wp in wyckoffs:
        wp.get_site_symmetry()
    site_symmetries = [wp.site_symm for wp in wyckoffs]
    site_enumeration = [wychoffs_enumerated_by_ss[pyxtal_structure.group.number][wp.letter] for wp in wyckoffs]
    multiplicity = [wp.multiplicity for wp in wyckoffs]
    dof = [wp.get_dof() for wp in wyckoffs]

    order = np.lexsort((site_enumeration, multiplicity, electronegativity))

    sites_dict = {
        "site_symmetries": [site_symmetries[i] for i in order],
        "elements": [elements[i] for i in order],
        "multiplicity": [multiplicity[i] for i in order],
        "wyckoff_letters": [wyckoffs[i].letter for i in order],
        "sites_enumeration": [site_enumeration[i] for i in order],
        "dof": [dof[i] for i in order],
        "spacegroup_number": pyxtal_structure.group.number
    }
    if wychoffs_augmentation is not None:
        augmented_enumeration = [
            [wychoffs_enumerated_by_ss[pyxtal_structure.group.number][augmentator[letter]] for
              letter in sites_dict["wyckoff_letters"]]
                for augmentator in wychoffs_augmentation[pyxtal_structure.group.number]
        ]
        sites_dict["sites_enumeration_augmented"] = augmented_enumeration
    return sites_dict


def read_MP(MP_csv: Path|str):
    """
    Reads a Materials Project CSV file and returns a DataFrame with structures.

    Args:
        MP_csv (Path|str): The path to the Materials Project CSV file.

    Returns:
        pd.DataFrame: The DataFrame with structures.
    """
    MP_df = pd.read_csv(MP_csv, index_col=0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Issues encountered while parsing CIF: \d+"
                " fractional coordinates rounded to ideal"
                " values to avoid issues with finite precision.",
            category=UserWarning,
            module="pymatgen.io.cif"
        )
        print("Suppressed CIF rounding warnings.")
        with Pool() as pool:
            MP_df["structure"] = pool.map(read_cif, MP_df["cif"])
    MP_df.drop(columns=["cif"], inplace=True)
    return MP_df


def compute_symmetry_sites(
    datasets_pd: dict[str, pd.DataFrame],
    wychoffs_enumerated_by_ss_file: Path = Path(__file__).parent.resolve() / "wychoffs_enumerated_by_ss.pkl.gz"
    ) -> tuple[dict[str, pd.DataFrame], int]:

    with open(wychoffs_enumerated_by_ss_file, "rb") as f:
        wychoffs_enumerated_by_ss = pickle.load(f)[0]
    
    structure_to_sites_with_args = partial(
                structure_to_sites,
                wychoffs_enumerated_by_ss=wychoffs_enumerated_by_ss,
                wychoffs_augmentation=get_augmentation_dict()
        )

    result = {}
    for dataset_name, dataset in datasets_pd.items():
        with Pool() as p:
            symmetry_dataset = pd.DataFrame.from_records(
                p.map(structure_to_sites_with_args, dataset['structure'])).set_index(dataset.index)
        result[dataset_name] = symmetry_dataset
    return result


def read_all_MP_csv(
    mp_path: Path = Path(__file__).parent.resolve() / "cdvae"/"data"/"mp_20",
    wychoffs_enumerated_by_ss_file: Path = Path(__file__).parent.resolve() / "wychoffs_enumerated_by_ss.pkl.gz",
    file_format: str = "csv"
    ) -> tuple[dict[str, pd.DataFrame], int]:
    """
    Reads all Materials Project CSV files and returns a dictionary of DataFrames.

    Args:
        mp_path (Path, optional): The path to the Materials Project CSV files. Defaults to "cdvae/data/mp_20".
        wychoffs_enumerated_by_ss_file (Path, optional): The path to the Wyckoff positions enumerated by space group file. Defaults to "wychoffs_enumerated_by_ss.pkl.gz".
        file_format (str, optional): The file format. Defaults to "csv". Can be archived csv openable by pandas.

    Returns:
        dict: A dictionary with the following keys:
            - train: DataFrame with training data
            - test: DataFrame with testing data
            - val: DataFrame with validation data
    """
    datasets_pd = {
        "train": read_MP(mp_path / f"train.{file_format}"),
        "test": read_MP(mp_path / f"test.{file_format}"),
        "val": read_MP(mp_path / f"val.{file_format}")
    }
    symmetry_datasets = compute_symmetry_sites(datasets_pd, wychoffs_enumerated_by_ss_file)
    return symmetry_datasets


def read_mp_ternary_csv(
    mp_ternary_path: Path = Path(__file__).parent.resolve() / "cache" / "mp_ternary" / "df_allternary_newdata.csv.gz",
    wychoffs_enumerated_by_ss_file: Path = Path(__file__).parent.resolve() / "wychoffs_enumerated_by_ss.pkl.gz"
    ) -> tuple[dict[str, pd.DataFrame], int]:
    all_data = read_MP(mp_ternary_path)
    train, test_validation = train_test_split(all_data, train_size=0.6, random_state=42)
    validation, test = train_test_split(test_validation, train_size=0.5, random_state=43)
    datasets_pd = {
        "train": train,
        "test": test,
        "val": validation
    }
    symmetry_datasets = compute_symmetry_sites(datasets_pd, wychoffs_enumerated_by_ss_file)
    return datasets_pd

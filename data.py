"""
This module provides functions for reading the structures and extracting the symmetry information.
"""

from itertools import chain
from operator import itemgetter
from pathlib import Path
import logging
import warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure, Element, DummySpecie
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal


def conventional_standard_structure_from_cif(cif: str, symprec=0.1) -> Structure:
    """
    Reads a CIF file and returns the conventional standard structure.

    Args:
        cif (str): The CIF file content.
        symprec (float, optional): The symmetry precision. Defaults to 0.1.

    Returns:
        Structure: The conventional standard structure.
    """
    raw_structure = CifParser.from_str(cif).get_structures()[0]
    return SpacegroupAnalyzer(raw_structure, symprec=symprec).get_conventional_standard_structure()


def read_cif(cif: str) -> Structure:
    """
    Reads a CIF file and returns the structure.

    Args:
        cif (str): The CIF file content.

    Returns:
        Structure: The structure.
    """
    return CifParser.from_str(cif).get_structures()[0]


def structure_to_sites(structure: Structure, tol: float = 0.1) -> dict:
    """
    Converts a pymatgen structure to a dictionary of symmetry sites.

    Args:
        structure (Structure): The pymatgen structure.
        tol (float, optional): The tolerance. Defaults to 0.1.

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
    multiplicity = [wp.multiplicity for wp in wyckoffs]

    order = np.lexsort((multiplicity, electronegativity))

    return {
        "symmetry_sites": [site_symmetries[i] for i in order],
        "symmetry_elements": [elements[i] for i in order],
        "spacegroup_number": pyxtal_structure.group.number,
    }


def read_MP(MP_csv: Path|str):
    """
    Reads a Materials Project CSV file and returns a DataFrame with structures.

    Args:
        MP_csv (Path|str): The path to the Materials Project CSV file.

    Returns:
        pd.DataFrame: The DataFrame with structures.
    """
    MP_df = pd.read_csv(MP_csv, index_col=0)
    warnings.warn("Fractional coordinates will be silently rounded to ideal values to avoid issues with finite precision")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=r".*fractional coordinates rounded to ideal values to"
                                "avoid issues with finite precision")
        with Pool() as pool:
            MP_df["structure"] = pool.map(read_cif, MP_df["cif"])
    MP_df.drop(columns=["cif"], inplace=True)
    return MP_df


def read_all_MP_csv(mp20_path=Path(__file__).parent.resolve() / "cdvae"/"data"/"mp_20"):
    """
    Reads all Materials Project CSV files and returns a dictionary of DataFrames.

    Args:
        mp20_path (Path, optional): The path to the Materials Project CSV files. Defaults to "cdvae/data/mp_20".

    Returns:
        dict: A dictionary with the following keys:
            - train: DataFrame with training data
            - test: DataFrame with testing data
            - val: DataFrame with validation data
    """
    datasets_pd = {
        "train": read_MP(mp20_path/"train.csv"),
        "test": read_MP(mp20_path/"test.csv"),
        "val": read_MP(mp20_path/"val.csv")
    }
    for dataset in datasets_pd.values():
        with Pool() as p:
            symmetry_dataset = pd.DataFrame.from_records(
                p.map(structure_to_sites, dataset['structure'])).set_index(dataset.index)
        dataset.loc[:, symmetry_dataset.columns] = symmetry_dataset

    max_len = max(map(len, chain.from_iterable(map(itemgetter("symmetry_sites"), datasets_pd.values())))) + 1
    for dataset in datasets_pd.values():
        dataset['symmetry_sites_padded'] = [x + ["STOP"] + ["PAD"] * (max_len - len(x)) for x in dataset['symmetry_sites']]
        dataset['symmetry_elements_padded'] = [x + ["STOP"] + [DummySpecie()] * (max_len - len(x)) for x in dataset['symmetry_elements']]
        dataset['padding_mask'] = [[False] * (len(x) + 1) + [True] * (max_len - len(x)) for x in dataset['symmetry_sites']]
    logging.info("Max length of symmetry sites: %i", max_len)
    return datasets_pd, max_len

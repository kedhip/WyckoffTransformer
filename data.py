"""
This module provides functions for reading the structures and extracting the symmetry information.
"""

from itertools import chain, repeat
from operator import itemgetter
from pathlib import Path
import logging
import pickle
import gzip
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from sklearn.model_selection import train_test_split


def conventional_standard_structure_from_cif(cif: str, symprec=0.1) -> Structure:
    """
    Reads a CIF file and returns the conventional standard structure.

    Args:
        cif (str): The CIF file content.
        symprec (float, optional): The symmetry precision. Defaults to 0.1.

    Returns:
        Structure: The conventional standard structure.
    """
    raw_structure = CifParser.from_str(cif).parse_structures(primitive=False)[0]
    return SpacegroupAnalyzer(raw_structure, symprec=symprec).get_conventional_standard_structure()


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
    tol: float = 0.1) -> dict:
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
    site_enumeration = [wychoffs_enumerated_by_ss[pyxtal_structure.group.number][wp.letter] for wp in wyckoffs]
    multiplicity = [wp.multiplicity for wp in wyckoffs]
    dof = [wp.get_dof() for wp in wyckoffs]

    order = np.lexsort((site_enumeration, multiplicity, electronegativity))

    return {
        "symmetry_sites": [site_symmetries[i] for i in order],
        "symmetry_elements": [elements[i] for i in order],
        "symmetry_multiplicity": [multiplicity[i] for i in order],
        "symmetry_letters": [wyckoffs[i].letter for i in order],
        "symmetry_sites_enumeration": [site_enumeration[i] for i in order],
        "symmetry_dof": [dof[i] for i in order],
        "spacegroup_number": pyxtal_structure.group.number
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
    
    for dataset in datasets_pd.values():
        with Pool() as p:
            symmetry_dataset = pd.DataFrame.from_records(
                p.starmap(structure_to_sites, zip(dataset['structure'], repeat(wychoffs_enumerated_by_ss)))).set_index(dataset.index)
        dataset.loc[:, symmetry_dataset.columns] = symmetry_dataset

    # TODO investigate the "long" materials
    MAX_LEN_LIMIT = 21
    for dataset_name in datasets_pd:
        datasets_pd[dataset_name] = datasets_pd[dataset_name].loc[datasets_pd[dataset_name]['symmetry_sites'].map(len) < MAX_LEN_LIMIT]
    max_len = max(map(len, chain.from_iterable(map(itemgetter("symmetry_sites"), datasets_pd.values())))) + 1
    for dataset in datasets_pd.values():
        dataset.loc[:, "lattice_volume"] = [s.get_primitive_structure().lattice.volume for s in dataset['structure']]
        dataset.loc[:, 'symmetry_sites_padded'] = [x + ["PAD"] * (max_len - len(x) + 1) for x in dataset['symmetry_sites']]
        dataset.loc[:, 'symmetry_elements_padded'] = [x + ["STOP"] + ["PAD"] * (max_len - len(x)) for x in dataset['symmetry_elements']]
        dataset.loc[:, 'symmetry_sites_enumeration_padded'] = [x + [-1] * (max_len - len(x) + 1) for x in dataset['symmetry_sites_enumeration']]
        dataset.loc[:, 'padding_mask'] = [[False] * (len(x) + 1) + [True] * (max_len - len(x)) for x in dataset['symmetry_sites']]
    logging.info("Max length of symmetry sites: %i", max_len)
    return datasets_pd, max_len


def read_all_MP_csv(
    mp20_path: Path = Path(__file__).parent.resolve() / "cdvae"/"data"/"mp_20",
    wychoffs_enumerated_by_ss_file: Path = Path(__file__).parent.resolve() / "wychoffs_enumerated_by_ss.pkl.gz"
    ) -> tuple[dict[str, pd.DataFrame], int]:
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
    return compute_symmetry_sites(datasets_pd, wychoffs_enumerated_by_ss_file)


def read_mp_ternary_csv(
    mp_ternary_path: Path = Path(__file__).parent.resolve() / "cache" / "mp_ternary" / "df_allternary_newdata.csv.gz",
    wychoffs_enumerated_by_ss_file: Path = Path(__file__).parent.resolve() / "wychoffs_enumerated_by_ss.pkl.gz"
    ) -> tuple[dict[str, pd.DataFrame], int]:
    all_data = read_MP(mp_ternary_path)
    train, test_validation = train_test_split(all_data, train_size=27136, random_state=42)
    # TODO(kazeevn) this is embarassing. We really need to batch.
    validation, test = train_test_split(test_validation, train_size=0.5, random_state=43)
    datasets_pd = {
        "train": train,
        "test": test,
        "val": validation
    }
    return compute_symmetry_sites(datasets_pd, wychoffs_enumerated_by_ss_file)

from itertools import chain
from operator import itemgetter
from pathlib import Path
import logging
from multiprocessing import Pool
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from collections import Counter
import numpy as np
import pandas as pd
from pymatgen.core import DummySpecie
from pyxtal import pyxtal


def conventional_standard_structure_from_cif(cif: str) -> Structure:
    raw_structure = CifParser.from_str(cif).get_structures()[0]
    return SpacegroupAnalyzer(raw_structure).get_conventional_standard_structure()


def read_one_MP(MP_csv: Path|str) -> pd.DataFrame:
    MP_df = pd.read_csv(MP_csv, index_col=0)
    with Pool() as pool:
        MP_df["structure"] = pool.map(
            conventional_standard_structure_from_cif, MP_df["cif"])
    MP_df.drop(columns=["cif"], inplace=True)
    return MP_df


def structure_to_sites(structure: Structure) -> dict:
    # Note(kazeevn):
    # We lose information about coordinates (as expected)
    # We also can't disambiguate between different orbits with same site symmetry
    # while the absolute coordinates might depend wholly on the unit cell choice (TBC), relatively to each other they are not
    pyxtal_structure = pyxtal()
    # Tolerance per https://pymatgen.org/pymatgen.symmetry.html#pymatgen.symmetry.analyzer.SpacegroupAnalyzer
    # For structures with slight deviations from their proper atomic positions
    # (e.g., structures relaxed with electronic structure codes), a looser tolerance
    # of 0.1 (the value used in Materials Project) is often needed.
    pyxtal_structure.from_seed(structure, tol=0.1)
    
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


def structure_from_cif(cif: str):
    raw_structure = CifParser.from_str(cif).get_structures()[0]
    return SpacegroupAnalyzer(raw_structure, symprec=0.1).get_conventional_standard_structure()


def read_MP(MP_csv: Path|str):
    MP_df = pd.read_csv(MP_csv, index_col=0)
    with Pool() as pool:
        MP_df["structure"] = pool.map(structure_from_cif, MP_df["cif"])
    MP_df.drop(columns=["cif"], inplace=True)
    return MP_df

def read_all_MP_csv(mp20_path=Path(__file__).parent.resolve() / "cdvae"/"data"/"mp_20"):
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

    # Pad the lists to ease the batching
    # Maybe in the future we should use NestedTensor or EncoderLayer's buit-in padding

    max_len = max(map(len, chain.from_iterable(map(itemgetter("symmetry_sites"), datasets_pd.values())))) + 1
    for dataset in datasets_pd.values():
        dataset['symmetry_sites_padded'] = [x + ["STOP"] + ["PAD"] * (max_len - len(x)) for x in dataset['symmetry_sites']]
        dataset['symmetry_elements_padded'] = [x + ["STOP"] + [DummySpecie()] * (max_len - len(x)) for x in dataset['symmetry_elements']]
        dataset['padding_mask'] = [[False] * (len(x) + 1) + [True] * (max_len - len(x)) for x in dataset['symmetry_sites']]
    logging.info(f"Max length of symmetry sites: {max_len}")
    return datasets_pd, max_len
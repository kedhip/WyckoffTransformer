from enum import Enum
from typing import Optional
import warnings
import gzip
from multiprocessing import Pool
import pickle
from pymatgen.core import Composition
from pathlib import Path
from ast import literal_eval
import monty.json
import pandas as pd
import sys
sys.path.append("..")
from data import read_cif, compute_symmetry_sites, read_MP


from .DiffCSP_to_sites import load_diffcsp_dataset, record_to_pyxtal
from .cdvae.refined_metrics import Crystal

Transformation = Enum("Transformation", [
    "WyckoffTransformer",
    "WyCryst",
    "DiffSCP++",
    "DiffCSP",
    "CrystalFormer",
    "CHGNet_fix_symmetry",
    "CHGNet_relax_symmetry"])

StructureStorage = Enum("StructureStorage", [
    "DiffCSP_pt",
    "CrystalFormer",
    "NongWei",
    "Raymond",
    "CDVAE_csv_cif"
    ])

WyckoffStorage = Enum("WyckoffStorage", [
    "pyxtal_json",
    "WTCache"])

class GenerationPipeline():
    """
    Our structure generation and evaluation pipeline has a lot of variants.
    1. Obtain Wyckoff positions 
    1.a Generate [WyckoffTransformer, WyCryst]
    1.b Get from a template [DiffSCP++]
    2. Generate structures that are considered "model outputs"
    2.1 WyckoffTransformer, WyCryst + pyXtal + CHGNet
        Sample with pyxtal.from_random
        Relax with symmetry-constrained CHGNet
        Relax with CHGNet without symmetry constraints
    2.2 WyckoffTransformer, WyCryst + DiffSCP++
        Used as a template for DiffSCP++
    2.3 WyckoffTransformer, WyCryst + DiffSCP++ + CHGNet
        Used as a template for DiffSCP++
        Relax with symmetry-constrained CHGNet
        Relax with CHGNet without symmetry constraints
    2.4 WyckoffTransformer, WyCryst + DiffSCP++ + CHGNet (no symmetry)
        Used as a template for DiffSCP++
        Relax with CHGNet without symmetry constraints
    2.3 CrystalFormer: generate together with the Wyckoff positions
    2.4 DiffCSP++: diffuse from the template
    2.5 DiffCSP and alike: de novo diffusion
    3. Compute E_hull: relax with CHGNet without symmetry constraints

    The class is designed around the following assumptions:
    a. Every generated dataset has the one and only one primary source:
        a list of either Wyckoff representations or structures.
    b. Transformations form a tree structure
    c. We want to compute with SpaceGroupAnalyzer and cache and wyckoff positions for structures

    Evaluations are of three types:
    1. (main) Finished structures vs finished structures
     - We should be generous with competing methods - evaluate both raw and CHGNet structures, choose the best
    2. Consistency along the transformation graph
    3. Intermediary structures vs intermediary structures

    Novelty/uniqueness is computed in two variants:
    1. Wyckoff positions
    2. StructureMatcher

    Structures must be indexed in the same way across all datasets, this allows for easy novelty/uniqueness masking.

    Evaluations to be performed:
    1. CDVAE metrics, to get them out of the way: finished structures, "validity" filtering, no novelty/uniqueness filtering
    2. S. U. N and S N (among U).
    3. Property distributions and stability, UN structures only
    4. SG preservation after relaxation, UN structures only

    Data are stored as a dictionary.
    {"transformation": Transformation, "data": GeneratedDataset, "children": Dict[Transformation, Dict]}
    """
def __init__(self):
    self.data_root = None


def load_crystalformer(path: Path):
    dataset = pd.read_csv(path)
    decoder = monty.json.MontyDecoder()
    structures = dataset.cif.map(lambda s: decoder.process_decoded(literal_eval(s)))
    structures.name = "structures"
    return structures


def load_NongWei(path: Path):
    # There are two possible directory structures:
    # 0-499 & 500-999 / 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ...
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ...
    # We handle them all by using rglob 
    data = pd.read_csv(path/"id_with_energy_per_atom.csv", index_col="id",
            usecols=["id", "energy_per_atom"])
    data['structure'] = pd.Series(dtype=object, index=data.index)
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
        cif_file_paths = path.rglob("**/min_e_strc.cif")
        for this_cif_path in cif_file_paths:
            with open(this_cif_path, "rt", encoding="ascii") as f:
                data.at[int(this_cif_path.parent.stem), "structure"] = read_cif(f.read())
    return data


def load_Raymond(path: Path):
    index_csv_path = list(path.glob("*.csv"))
    if len(index_csv_path) != 1:
        raise ValueError("No index CSV found")
    index_csv_path = index_csv_path[0]
    data = pd.read_csv(index_csv_path, index_col=0)
    data['structure'] = pd.Series(dtype=object, index=data.index)
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
        cif_file_paths = path.rglob("**/*relax_postitions_and_cell*.cif")
        for this_cif_path in cif_file_paths:
            with open(this_cif_path, "rt", encoding="ascii") as f:
                data.at[int(this_cif_path.parent.stem), "structure"] = read_cif(f.read())
    return data


class GeneratedDataset():
    WyckoffSource = Enum("WyckoffSource", ["pyxtal", "external"])

    # TODO: handle various missing data
    def __init__(self):
        self.data = pd.DataFrame()
        self.wyckoffs_source = None

    def load_structures(self, path: Path|str, storage_type: StructureStorage):
        # If structures are defined, wyckoffs must be obtained with pyxtal.from_seed(structures)
        # If structures are not defined, wyckoffs can be read externally
        if "wyckoffs" in self.data.columns:
            raise ValueError("Wyckoff positions are already defined")
        if isinstance(path, str):
            path = Path(path)

        if storage_type == StructureStorage.DiffCSP_pt:
            self.data["structure"] = load_diffcsp_dataset(path)
        elif storage_type == StructureStorage.CrystalFormer:
            self.data["structure"] = load_crystalformer(path)
        elif storage_type == StructureStorage.NongWei:
            self.data = load_NongWei(path)
        elif storage_type == StructureStorage.Raymond:
            self.data = load_Raymond(path)
        elif storage_type == StructureStorage.CDVAE_csv_cif:
            self.data = read_MP(path)
        else:
            raise ValueError("Unknown storage type")
        if self.data.index.duplicated().any():
            raise ValueError("Duplicate indices in the dataset")
        self.data.sort_index(inplace=True)

    def load_corrected_chgnet_ehull(self, path: Path|str):
        e_hull_data = pd.read_csv(path)
        if "folder_ind" in e_hull_data.columns:
            e_hull_data.set_index("folder_ind", inplace=True)
        else:
            e_hull_data.set_index(e_hull_data.columns[0], inplace=True)
        # Verify that the data matches
        e_hull_data = e_hull_data.reindex(self.data.index, copy=False)
        for e_hull_formula, structure in zip(e_hull_data["formula"], self.data["structure"]):
            if isinstance(e_hull_formula, float):
                continue
            if Composition(e_hull_formula).reduced_composition != structure.composition.reduced_composition:
                raise ValueError(f"Formula mismatch between {e_hull_formula} and {structure.composition}")
        self.data["corrected_chgnet_ehull"] = e_hull_data["corrected_chgnet_ehull"]

    def load_wyckoffs(self, path: Path|str, storage_type: WyckoffStorage, cache_key: Optional[str] = None):
        if storage_type == WyckoffStorage.pyxtal_json:
            wcykoffs = pd.read_json(path)
        elif storage_type == WyckoffStorage.WTCache:
            with gzip.open(path, "rb") as f:
                wcykoffs = pickle.load(f)[cache_key]
        else:
            raise ValueError("Unknown storage type")
        if len(self.data) == 0:
            self.data = wcykoffs
        else:
            self.data.loc[:, wcykoffs.columns] = wcykoffs
        self.wyckoffs_source = GeneratedDataset.WyckoffSource.external

    def compute_wyckoffs(self, n_jobs: Optional[int] = None):
        if self.wyckoffs_source == GeneratedDataset.WyckoffSource.external:
            raise ValueError("Wyckoffs are already defined externally")
        if "structure" not in self.data.columns:
            raise ValueError("Structures are not defined")
        wyckoffs = compute_symmetry_sites({"_": self.data}, n_jobs=n_jobs)["_"]
        self.data.loc[:, wyckoffs.columns] = wyckoffs
        self.wyckoffs_source = GeneratedDataset.WyckoffSource.pyxtal

    def convert_wyckoffs_to_pyxtal(self):
        in_pyxtal_format = pd.DataFrame.from_records(self.data.apply(record_to_pyxtal, axis=1))
        self.data.loc[:, in_pyxtal_format.columns] = in_pyxtal_format

    def compute_cdvae_crystals(self, n_jobs: Optional[int] = None):
        with Pool(n_jobs) as pool:
            self.data["cdvae_crystal"] = pool.map(Crystal.from_pymatgen, self.data["structure"])
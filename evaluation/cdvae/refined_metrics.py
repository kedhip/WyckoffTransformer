"""
Metrics defined in the CDVAE paper:
Xie, Tian, et al. "Crystal diffusion variational autoencoder for periodic material generation."
arXiv preprint arXiv:2110.06197 (2021).
Code from:
https://github.com/txie-93/cdvae
https://github.com/jiaor17/DiffCSP/
"""
from itertools import product
from collections import Counter
from math import gcd
from typing import Dict, Optional
import logging
import numpy as np
from wrapt_timeout_decorator import timeout
import smact.screening
from pandas import Series
from pymatgen.core import Composition, Structure
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

logger = logging.getLogger(__name__)

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

class Crystal(object):
    @classmethod
    def from_pymatgen(cls, structure: Structure):
        crys_array_dict = {
            'frac_coords': structure.frac_coords,
            'atom_types': np.array([s.number for s in structure.species]),
            'lengths': np.array(structure.lattice.abc),
            'angles': np.array(structure.lattice.angles)
        }
        return cls(crys_array_dict, structure)

    def __init__(self,
        crys_array_dict: Dict[str, np.ndarray],
        structure: Optional[Structure] = None):

        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
            self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)

        if structure is not None:
            self.structure = structure
            self.constructed = True
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()


    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = timed_smact_validity_from_composition(self.structure.composition)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            logger.debug("Fingerprint construction failed.")
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)



def smact_validity_optimised(
                   elem_symbols, count,
                   use_pauling_test=True,
                   include_alloys=True,
                   apply_gcd=False):
    element_set = frozenset(elem_symbols)
    if len(element_set) == 1:
        return True
    if include_alloys and element_set.issubset(smact.metals):
        return True
    
    space = smact.element_dictionary(elem_symbols)
    electronegs = [e.pauling_eneg for e in space.values()]
    ox_combos = [e.oxidation_states for e in space.values()]

    if apply_gcd:
        gcd_count = gcd(*count)
        count = tuple(c // gcd_count for c in count)
    threshold = max(count)
    stoichs = [(c,) for c in count]
    for ox_states in product(*ox_combos):
        cn_r = smact.neutral_ratios_iter(ox_states, stoichs=stoichs, threshold=threshold)
        if any(True for _ in cn_r):
            if not use_pauling_test:
                return True
            try:
                if smact.screening.pauling_test(ox_states, electronegs, symbols=elem_symbols):
                    return True
            except TypeError:
                # if no electronegativity data, assume it is okay
                return True
    return False


# Pandas tries to treat the decorated function as iterable, which causes a TypeError.
@timeout(15)
def timed_smact_validity_optimised(*args, **kwargs) -> bool:
    """
    Computes the SMACT validity of a record in pyxtal.from_random arguments format.
    If the computation takes longer than 15 seconds, returns False.
    """
    try:
        return smact_validity_optimised(*args, **kwargs)
    except TimeoutError:
        logger.debug("SMAC-T validity timed out, returning False")
        return False


def timed_smact_validity_from_composition(composition: Composition, apply_gcd: bool=True) -> bool:
    """
    Computes the SMACT validity of a record in pyxtal.from_random arguments format.
    If the computation takes longer than 15 seconds, returns False.
    """
    get_el_amt_dict = composition.get_el_amt_dict()
    return timed_smact_validity_optimised(
        elem_symbols=list(get_el_amt_dict.keys()),
        count=list(map(int, get_el_amt_dict.values())),
        apply_gcd=apply_gcd)


def timed_smact_validity_from_record(record: Dict|Series, apply_gcd: bool=True) -> bool:
    """
    Computes the SMACT validity of a record in pyxtal.from_random arguments format.
    If the computation takes longer than 15 seconds, returns False.
    """
    return timed_smact_validity_optimised(record["species"], record["numIons"], apply_gcd=apply_gcd)


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.full(dist_mat.shape[0], cutoff + 10.))
    if crystal.volume < 0.1 or dist_mat.min() < cutoff:
        return False
    return True

def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps

class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}


    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}


    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in self.valid_samples])
            gt_props = prop_model_eval(self.eval_model_name, [
                                       c.dict for c in self.gt_crys])
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}
        else:
            return {'wdist_prop': None}


def compute_cov(crys, gt_crys,
                struc_cutoff, comp_cutoff, num_gen_crystals=None):
    struc_fps = [c.struct_fp for c in crys]
    comp_fps = [c.comp_fp for c in crys]
    gt_struc_fps = [c.struct_fp for c in gt_crys]
    gt_comp_fps = [c.comp_fp for c in gt_crys]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)

    comp_fps = CompScaler.transform(comp_fps)
    gt_comp_fps = CompScaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    metrics_dict = {
        'cov_recall': cov_recall,
        'cov_precision': cov_precision,
        'amsd_recall': np.mean(struc_recall_dist),
        'amsd_precision': np.mean(struc_precision_dist),
        'amcd_recall': np.mean(comp_recall_dist),
        'amcd_precision': np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        'struc_recall_dist': struc_recall_dist.tolist(),
        'struc_precision_dist': struc_precision_dist.tolist(),
        'comp_recall_dist': comp_recall_dist.tolist(),
        'comp_precision_dist': comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict

from typing import Dict, Iterable
from itertools import chain, product
from collections import defaultdict
import numpy as np
from scipy.stats import kstest
import pandas as pd
from pyxtal.symmetry import Group
from pymatgen.core import Element
import smact
import smact.screening

def count_unique(x: Iterable) -> int:
    return len(frozenset(x))


class DoFCounter():
    def __init__(self):
        dof_from_letter = defaultdict(dict)
        for group_number in range(1, 231):
            group = Group(group_number)
            for wp in group.Wyckoff_positions:
                wp.get_site_symmetry()
                dof_from_letter[group_number][wp.letter] = wp.get_dof()
        self.dof_from_letter = dof_from_letter

    def get_total_dof(self, record: Dict) -> int:
        """
        Computes the total number of degrees of freedom in a structure.
        Args:
            record: dictionary with a structure in pyxtal.from_random arguments format.
        """
        return sum(map(lambda site: self.dof_from_letter[record['group']][site.lstrip('1234567890')],
            chain(*record['sites'])))


# Function from CDVAE
# https://github.com/txie-93/cdvae/blob/f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea/scripts/eval_utils.py
def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):
    elem_symbols = tuple(comp)
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    for ox_states in product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = smact.screening.pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(
                        tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False


def smact_validity_optimised(
                   elem_symbols, count,
                   use_pauling_test=True,
                   include_alloys=True):
    element_set = frozenset(elem_symbols)
    if len(element_set) == 1:
        return True
    if include_alloys and element_set.issubset(smact.metals):
        return True
    
    space = smact.element_dictionary(elem_symbols)
    electronegs = [e.pauling_eneg for e in space.values()]
    ox_combos = [e.oxidation_states for e in space.values()]

    threshold = max(count)
    stoichs = [(c,) for c in count]
    for ox_states in product(*ox_combos):
        cn_r = smact.neutral_ratios_iter(ox_states, stoichs=stoichs, threshold=threshold)
        if any(True for _ in cn_r):
            if not use_pauling_test:
                return True
            try:
                if smact.screening.pauling_test(ox_states, electronegs):
                    return True
            except TypeError:
                # if no electronegativity data, assume it is okay
                return True
    return False


def smact_validity_from_record(record: Dict) -> bool:
    """
    Computes the SMACT validity of a record in pyxtal.from_random arguments format.
    """
    return smact_validity_optimised(record['species'], record['numIons'])


class StatisticalEvaluator():
    def __init__(self,
                 test_dataset: pd.DataFrame):
        self.test_dataset = test_dataset
        self.dof_counter = None

    def get_num_sites_ks(self, generated_structures: Iterable[Dict], return_counts:bool=False) -> float:
        """
        Computes the Kolmogorov-Smirnov statistic between the numbers of sites in the 
        generated structures and the test dataset. Note that Kolmorogov-Smirnov statistic doesn't
        a priory depend on the number examples, but p-value does.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
        """
        test_num_sites = self.test_dataset['site_symmetries'].map(len)
        generated_num_sites = [sum(map(len, wygene['sites'])) for wygene in generated_structures]
        if return_counts:
            return kstest(test_num_sites, generated_num_sites), generated_num_sites
        else:
            return kstest(test_num_sites, generated_num_sites)


    def get_num_elements_ks(self, generated_structures: Iterable[Dict], return_counts:bool=False) -> float:
        """
        Computes the Kolmogorov-Smirnov statistic between the numbers of elements in the 
        generated structures and the test dataset. Note that Kolmorogov-Smirnov statistic doesn't
        a priory depend on the number examples, but p-value does.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
        """
        test_num_elements = self.test_dataset['elements'].map(count_unique)
        generated_num_elements = [len(frozenset(record["species"])) for record in generated_structures]
        if return_counts:
            return kstest(test_num_elements, generated_num_elements), generated_num_elements
        else:
            return kstest(test_num_elements, generated_num_elements)


    def get_dof_ks(self, generated_structures: Iterable[Dict]) -> float:
        """
        Computes the Kolmogorov-Smirnov statistic between the degrees of freedom in the 
        generated structures and the test dataset. Note that Kolmorogov-Smirnov statistic doesn't
        a priory depend on the number examples, but p-value does.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
        """
        if self.dof_counter is None:
            self.dof_counter = DoFCounter()
        test_dof = self.test_dataset['dof'].apply(sum)
        generated_dof = [self.dof_counter.get_total_dof(record) for record in generated_structures]
        return kstest(test_dof, generated_dof)


def smac_validity_from_counter(counter: Dict[Element, float]) -> bool:
    return smact_validity_optimised(tuple((elem.symbol for elem in counter.keys())), tuple(map(int, counter.values())))


def main():
    print("Loading data")
    import gzip, json
    with gzip.open("runs/l82pxye5/generated_wp.json.gz", "rb") as f:
        generated_structures = json.load(f)
    print("Testing naive and optimised SMACT validity functions")
    optimised_validity = list(map(smact_validity_from_record, generated_structures))
    naive_validity = list(map(smact_validity, (record['species'] for record in generated_structures), (record['numIons'] for record in generated_structures)))
    assert naive_validity == optimised_validity
    print(sum(optimised_validity) / len(generated_structures))
    with gzip.open("cache/mp_20_biternary/data.pkl.gz", "rb") as f:
        dataset_pd = pd.read_pickle(f)
    test_dataset = dataset_pd['test']
    # Compute SMACT for the test dataset
    test_smact_valid = test_dataset.composition.map(smac_validity_from_counter)
    test_smact_valid_naive = test_dataset.composition.map(lambda counter: smact_validity(tuple((elem.symbol for elem in counter.keys())), tuple(map(int, counter.values()))))
    assert test_smact_valid_naive.equals(test_smact_valid)
    print(test_smact_valid.sum() / len(test_smact_valid))
    print("Asserts passed")


if __name__ == "__main__":
    main()    
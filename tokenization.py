from itertools import chain
from operator import itemgetter
from collections import defaultdict
import torch
import pyxtal

PAD_TOKEN = "PAD"
STOP_TOKEN = "STOP"
MASK_TOKEN = "MASK"

def get_tokens(datasets_pd, dtype=torch.uint8):
    all_sites = set(chain.from_iterable(chain(*map(itemgetter("symmetry_sites"), datasets_pd.values()))))
    all_sites.update((PAD_TOKEN, STOP_TOKEN, MASK_TOKEN))
    all_elements = set(chain.from_iterable(chain(*map(itemgetter("symmetry_elements"), datasets_pd.values()))))
    all_elements.update((PAD_TOKEN, STOP_TOKEN, MASK_TOKEN))
    all_spacegroups = set(chain(*map(itemgetter("spacegroup_number"), datasets_pd.values())))
    # We assume that enumeration goes from 0 to max_enumeration - 1
    max_enumeration = max(map(max, chain.from_iterable(map(itemgetter("symmetry_sites_enumeration"), datasets_pd.values()))))
    assert max_enumeration + 2 < torch.iinfo(dtype).max
    enumeration_stop = max_enumeration + 1
    enumeration_pad = max_enumeration + 2
    assert len(all_sites) < torch.iinfo(dtype).max
    assert len(all_elements) < torch.iinfo(dtype).max
    assert len(all_spacegroups) < torch.iinfo(dtype).max
    # It's the true len, without start or stop
    max_len = max(map(len, chain.from_iterable(map(itemgetter("symmetry_sites"), datasets_pd.values()))))

    site_to_ids = {word: idx for idx, word in enumerate(all_sites)}
    element_to_ids = {word: idx for idx, word in enumerate(all_elements)}
    spacegroup_to_ids = {word: idx for idx, word in enumerate(all_spacegroups)}
    
    site_stop = [site_to_ids[STOP_TOKEN]]
    site_pad = [site_to_ids[PAD_TOKEN]]
    def sites_to_tensor(sites):
        return torch.tensor([site_to_ids[site] for site in sites] + site_stop + site_pad * (max_len - len(sites)), dtype=dtype)
    
    element_stop = [element_to_ids[STOP_TOKEN]]
    element_pad = [element_to_ids[PAD_TOKEN]]
    def element_to_tensor(elements):
        return torch.tensor(
            [element_to_ids[element] for element in elements] + element_stop + element_pad * (max_len - len(elements)),
             dtype=dtype)
    
    def spacegroup_to_tensor(spacegroup):
        return torch.tensor(spacegroup_to_ids[spacegroup], dtype=dtype)
    
    def enumeration_to_tensor(enumeration):
        return torch.tensor(
            enumeration + [enumeration_stop] + [enumeration_pad] * (max_len - len(enumeration)), dtype=dtype)

    tensors = defaultdict(dict)
    for dataset_name, dataset in datasets_pd.items():
        tensors[dataset_name]['symmetry_sites'] = torch.stack(dataset['symmetry_sites'].map(sites_to_tensor).to_list())
        tensors[dataset_name]['elements'] = torch.stack(dataset['symmetry_elements'].map(element_to_tensor).to_list())
        tensors[dataset_name]['symmetry_sites_enumeration'] = torch.stack(dataset['symmetry_sites_enumeration'].map(enumeration_to_tensor).to_list())
        tensors[dataset_name]['spacegroup_number'] = torch.stack(dataset['spacegroup_number'].map(spacegroup_to_tensor).to_list())
        # tensors[dataset_name]['padding_mask_tensor'] = dataset['padding_mask'].map(torch.tensor)
    return tensors, site_to_ids, element_to_ids, spacegroup_to_ids, max_len, max_enumeration, enumeration_stop, enumeration_pad


def get_enum_augmentation_tensor(
    spacegroup_to_ids: dict,
    site_to_ids: dict,
    wychoffs_enumerated_by_ss: dict,
    ss_from_letter: dict):
    # Target data structure:
    # alternatives_tensor[sg_id, alternative_id, ss, old_enum] = new_enum
    # [4, 1, [0, 1, 2], [0, 0, 0]]
    # alternatives = [x.split() for x in g.get_alternatives()['Transformed WP']]
    # site_enumeration = [wychoffs_enumerated_by_ss[pyxtal_structure.group.number][wp.letter] for wp in wyckoffs]
    # alternatives_enum
    alternatives_by_sg = [None] * len(spacegroup_to_ids)
    max_enum = 7
    for spacegroup_real, spacegroup_id in spacegroup_to_ids.items():
        alternatives_letters = [x.split() for x in pyxtal.Group(spacegroup_real).get_alternatives()['Transformed WP']]
        # print(alternatives_letters)
        reference_order = alternatives_letters[0]
        # Discard WPs that are not present in the dataset
        is_present = {}
        for letter in reference_order:
            ss = ss_from_letter[spacegroup_real][letter]
            if ss in site_to_ids:
                is_present[ss] = True
            else:
                is_present[ss] = False
        
        present_letters = frozenset(filter(lambda letter: is_present[ss_from_letter[spacegroup_real][letter]], reference_order))

        reference_filtered = [letter for letter in reference_order if letter in present_letters]

        enum_index = [wychoffs_enumerated_by_ss[spacegroup_real][letter] for letter in reference_filtered]
        ss_index = [site_to_ids[ss_from_letter[spacegroup_real][letter]] for letter in reference_filtered]
        index = torch.tensor([ss_index, enum_index])
        alternative_tensors = []
        for this_alternative in alternatives_letters:
            alternatives_enum = [wychoffs_enumerated_by_ss[spacegroup_real][letter] for letter in this_alternative
                if letter in present_letters]
            alternative_tensors.append(torch.sparse_coo_tensor(index, alternatives_enum))
        alternatives_by_sg[spacegroup_id] = torch.stack(alternative_tensors)
    return alternatives_by_sg
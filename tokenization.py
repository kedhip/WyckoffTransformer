from itertools import chain
from operator import itemgetter
from collections import defaultdict
import torch


MASK_SITE = "MASK_SITE"


def get_tokens(datasets_pd):
    all_sites = set(chain.from_iterable(chain(*map(itemgetter("symmetry_sites_padded"), datasets_pd.values()))))
    all_elements = set(chain.from_iterable(chain(*map(itemgetter("symmetry_elements_padded"), datasets_pd.values()))))
    all_spacegroups = set(chain(*map(itemgetter("spacegroup_number"), datasets_pd.values())))
    # In order to make conditional prediction of sites on elements, we need to make a predicion of site by
    # N sites, and (N+1) elements
    all_sites.add(MASK_SITE)

    site_to_ids = {word: idx for idx, word in enumerate(all_sites)}
    element_to_ids = {word: idx for idx, word in enumerate(all_elements)}
    spacegroup_to_ids = {word: idx for idx, word in enumerate(all_spacegroups)}
    
    def sites_to_tensor(sites):
        return torch.tensor([site_to_ids[site] for site in sites])
    
    def element_to_tensor(elements):
        return torch.tensor([element_to_ids[element] for element in elements])
    
    def spacegroup_to_tensor(spacegroup):
        return torch.tensor(spacegroup_to_ids[spacegroup])
    
    tensors = defaultdict(dict)
    for dataset_name, dataset in datasets_pd.items():
        tensors[dataset_name]['symmetry_sites_tensor'] = dataset['symmetry_sites_padded'].map(sites_to_tensor)
        tensors[dataset_name]['symmetry_elements_tensor'] = dataset['symmetry_elements_padded'].map(element_to_tensor)
        tensors[dataset_name]['spacegroup_number_tensor'] = dataset['spacegroup_number'].map(spacegroup_to_tensor)
        tensors[dataset_name]['padding_mask_tensor'] = dataset['padding_mask'].map(torch.tensor)
        tensors[dataset_name]['lattice_volume_tensor'] = dataset['lattice_volume'].map(torch.tensor)
    return tensors, site_to_ids, element_to_ids, spacegroup_to_ids
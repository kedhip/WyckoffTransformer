import pickle
import logging
import gzip
import torch
from data import read_all_MP_csv
from tokenization import get_tokens

MP_20_cache_data = "cache/mp_20/data.pkl.gz"
MP_20_cache_tensors = "cache/mp_20/tensors.pkl.gz"
MP_20_path = "cdvae/data/mp_20"


def load_all_data(device:str="cpu", allow_retokenize:bool=False):
    try:
        with gzip.open(MP_20_cache_data, "rb") as f:
            datasets_pd, max_len = pickle.load(f)
    except Exception as e:
        logging.info("Error reading data cache:")
        logging.info(e)
        logging.info("Reading from csv")
        datasets_pd, max_len = read_all_MP_csv()
        with gzip.open(MP_20_cache_data, "wb") as f:
            pickle.dump((datasets_pd, max_len), f)
    try:
        with gzip.open(MP_20_cache_tensors, "rb") as f:
            tensors, site_to_ids, element_to_ids, spacegroup_to_ids = pickle.load(f)
    except Exception as e:
        if not allow_retokenize:
            raise e
        logging.warning("Error reading tensor cache! The token order will change!")
        logging.info(e)
        logging.info("Generating new tensors")
        tensors, site_to_ids, element_to_ids, spacegroup_to_ids = get_tokens(datasets_pd)
        with gzip.open(MP_20_cache_tensors, "wb") as f:
            pickle.dump((tensors, site_to_ids, element_to_ids, spacegroup_to_ids), f)
    def to_combined_dataset(dataset):
        return dict(
            symmetry_sites=torch.stack(dataset['symmetry_sites_tensor'].to_list()).to(device),
            symmetry_elements=torch.stack(dataset['symmetry_elements_tensor'].to_list()).to(device),
            spacegroup_number=torch.stack(dataset['spacegroup_number_tensor'].to_list()).to(device),
            padding_mask = torch.stack(dataset['padding_mask_tensor'].to_list()).to(device),
        )
    torch_datasets = dict(zip(tensors.keys(), map(to_combined_dataset, tensors.values())))
    for dataset_name, dataset in torch_datasets.items():
        dataset["lattice_volume"] = torch.stack(datasets_pd[dataset_name]['lattice_volume'].map(torch.tensor).to_list()).to(device)
    return datasets_pd, torch_datasets, site_to_ids, element_to_ids, spacegroup_to_ids, max_len
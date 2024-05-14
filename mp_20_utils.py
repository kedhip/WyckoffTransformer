import pickle
from pathlib import Path
import logging
import gzip
import torch
import argparse
import os
from data import read_all_MP_csv, read_mp_ternary_csv
from tokenization import get_tokens

cache_folder = Path("cache")
cache_data = "data.pkl.gz"
cache_tensors = "tensors.pkl.gz"


def load_all_data(
    device:str="cpu",
    allow_retokenize:bool=False,
    dataset:str = "mp_20"):
    cache_data_file_name = cache_folder / dataset / cache_data
    try:
        with gzip.open(cache_data_file_name, "rb") as f:
            datasets_pd, max_len = pickle.load(f)
    except Exception as e:
        logging.info("Error reading data cache:")
        logging.info(e)
        logging.info("Reading from csv")
        if dataset == "mp_20":
            datasets_pd, max_len = read_all_MP_csv()
        elif dataset == "mp_ternary":
            datasets_pd, max_len = read_mp_ternary_csv()
        with gzip.open(cache_data_file_name, "wb") as f:
            pickle.dump((datasets_pd, max_len), f)
    cache_tensors_file_name = cache_folder / dataset / cache_tensors
    try:
        with gzip.open(cache_tensors_file_name, "rb") as f:
            tensors, site_to_ids, element_to_ids, spacegroup_to_ids = pickle.load(f)
    except Exception as e:
        if not allow_retokenize:
            raise e
        logging.warning("Error reading tensor cache! The token order will change!")
        logging.info(e)
        logging.info("Generating new tensors")
        tensors, site_to_ids, element_to_ids, spacegroup_to_ids = get_tokens(datasets_pd)
        with gzip.open(cache_tensors_file_name, "wb") as f:
            pickle.dump((tensors, site_to_ids, element_to_ids, spacegroup_to_ids), f)
    def to_combined_dataset(dataset):
        return dict(
            symmetry_sites=torch.stack(dataset['symmetry_sites_tensor'].to_list()).to(device),
            symmetry_elements=torch.stack(dataset['symmetry_elements_tensor'].to_list()).to(device),
            symmetry_sites_enumeration=torch.stack(dataset['symmetry_sites_enumeration_tensor'].to_list()).to(device),
            spacegroup_number=torch.stack(dataset['spacegroup_number_tensor'].to_list()).to(device),
            padding_mask = torch.stack(dataset['padding_mask_tensor'].to_list()).to(device),
        )
    torch_datasets = dict(zip(tensors.keys(), map(to_combined_dataset, tensors.values())))
    for dataset_name, dataset in torch_datasets.items():
        dataset["lattice_volume"] = torch.stack(datasets_pd[dataset_name]['lattice_volume'].map(torch.tensor).to_list()).to(device)
    return datasets_pd, torch_datasets, site_to_ids, element_to_ids, spacegroup_to_ids, max_len


def main():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="The device to use.")
    parser.add_argument("--allow_retokenize", action="store_true", help="Allow retokenization.")
    parser.add_argument("dataset", default="mp_20", help="The dataset to try loading from cache, re-reading if can't.")
    args = parser.parse_args()
    load_all_data(args.device, args.allow_retokenize, args.dataset)


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    # We want to avoid messing with the environment variables in case we are used as a module.
    # The code is parallelised by structure
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "1"

import pickle
from pathlib import Path
import gzip
import torch
import argparse
from data import read_all_MP_csv, read_mp_ternary_csv
from tokenization import get_tokens

cache_folder = Path("cache")
cache_data = "data.pkl.gz"
cache_tensors = "tensors.pkl.gz"


def get_cache_data_file_name(dataset:str):
    return cache_folder / dataset / cache_data


def get_cache_tensors_file_name(dataset:str):
    return cache_folder / dataset / cache_tensors


def cache_dataset(dataset:str):
    if dataset == "mp_20":
        datasets_pd, max_len = read_all_MP_csv()
    elif dataset == "mp_ternary":
        datasets_pd, max_len = read_mp_ternary_csv()
    elif dataset == "mp_20_biternary":
        datasets_pd, max_len = read_all_MP_csv(
            Path(__file__).parent.resolve() / "data" / "mp_20_biternary",
            file_format="csv.gz")
    cache_data_file_name = get_cache_data_file_name(dataset)
    cache_data_file_name.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_data_file_name, "wb") as f:
        pickle.dump((datasets_pd, max_len), f)

    tensors, site_to_ids, element_to_ids, spacegroup_to_ids = get_tokens(datasets_pd)
    cache_tensors_file_name = get_cache_tensors_file_name(dataset)
    cache_tensors_file_name.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_tensors_file_name, "wb") as f:
        pickle.dump((tensors, site_to_ids, element_to_ids, spacegroup_to_ids), f)


def load_all_data(
    device:str="cpu",
    dataset:str = "mp_20"):
    cache_data_file_name = get_cache_data_file_name(dataset)
    with gzip.open(cache_data_file_name, "rb") as f:
        datasets_pd, max_len = pickle.load(f)
    
    cache_tensors_file_name = get_cache_tensors_file_name(dataset)
    with gzip.open(cache_tensors_file_name, "rb") as f:
        tensors, site_to_ids, element_to_ids, spacegroup_to_ids = pickle.load(f)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", default="mp_20", help="The dataset to cache.")
    args = parser.parse_args()
    cache_dataset(args.dataset)


if __name__ == "__main__":
    main()

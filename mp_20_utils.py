if __name__ == "__main__":
    # We want to avoid messing with the environment variables in case we are used as a module.
    # The code is parallelised by structure
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "1"

import pickle
from pathlib import Path
import gzip
import argparse
from data import read_all_MP_csv, read_mp_ternary_csv
from tokenization import get_tokens

cache_folder = Path("cache")

def get_cache_data_file_name(dataset:str):
    return cache_folder / dataset / "data.pkl.gz"


def get_cache_tensors_file_name(dataset:str):
    return cache_folder / dataset / "tensors.pkl.gz"


def cache_dataset(dataset:str):
    """
    Loads a dataset, tokenizes and caches it.
    """
    if dataset == "mp_20":
        datasets_pd = read_all_MP_csv()
    elif dataset == "mp_ternary":
        datasets_pd = read_mp_ternary_csv()
    elif dataset == "mp_20_biternary":
        datasets_pd = read_all_MP_csv(
            Path(__file__).parent.resolve() / "data" / "mp_20_biternary",
            file_format="csv.gz")
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    cache_data_file_name = get_cache_data_file_name(dataset)
    cache_data_file_name.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_data_file_name, "wb") as f:
        pickle.dump(datasets_pd, f)


def cache_tensors(dataset:str):
    cache_data_file_name = get_cache_data_file_name(dataset)
    with gzip.open(cache_data_file_name, "rb") as f:
        datasets_pd = pickle.load(f)

    tensor_info = get_tokens(datasets_pd)
    cache_tensors_file_name = get_cache_tensors_file_name(dataset)
    cache_tensors_file_name.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_tensors_file_name, "wb") as f:
        pickle.dump(tensor_info, f)


def load_all_data(dataset:str):
    cache_data_file_name = get_cache_data_file_name(dataset)
    with gzip.open(cache_data_file_name, "rb") as f:
        datasets_pd = pickle.load(f)
    
    cache_tensors_file_name = get_cache_tensors_file_name(dataset)
    with gzip.open(cache_tensors_file_name, "rb") as f:
        tensor_info = pickle.load(f)
    
    return datasets_pd, *tensor_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="The dataset to cache.")
    parser.add_argument("--cache-dataset", action="store_true", help="Cache the dataset.")
    parser.add_argument("--cache-tensors", action="store_true", help="Cache the tensors.")
    args = parser.parse_args()
    if args.cache_dataset:
        cache_dataset(args.dataset)
    if args.cache_tensors:
        cache_tensors(args.dataset)


if __name__ == "__main__":
    main()

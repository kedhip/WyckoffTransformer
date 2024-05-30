import argparse
import omegaconf
from pathlib import Path
import pickle
import gzip

from wyckoff_transformer.tokenization import tokenise_dataset

def main():
    parser = argparse.ArgumentParser("Retokenise a cached dataset")
    parser.add_argument("dataset", type=str, help="The name of the dataset to retokenise")
    parser.add_argument("config_file", type=Path, help="The tokeniser configuration file")
    args = parser.parse_args()
    config = omegaconf.OmegaConf.load(args.config_file)
    cache_path = Path(__file__).parent.resolve() / "cache" / args.dataset
    cache_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path / 'data.pkl.gz', "rb") as f:
        datasets_pd = pickle.load(f)
    print("Loaded the dataset. It has the following sizes:")
    for name, dataset in datasets_pd.items():
        print(f"{name}: {len(dataset)}")
    tensors, tokenisers = tokenise_dataset(datasets_pd, config)
    tokeniser_name = args.config_file.stem
    cache_tensors_path = cache_path / 'tensors'
    cache_tensors_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_tensors_path / f'{tokeniser_name}.pkl.gz', "wb") as f:
        pickle.dump(tensors, f)
    # In the future we might want to save the tokenisers in json, so that they can be distributed
    cache_tokenisers_path = cache_path / 'tokenisers'
    cache_tokenisers_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_tokenisers_path / f'{tokeniser_name}.pkl.gz', "wb") as f:
        pickle.dump(tokenisers, f)


if __name__ == '__main__':
    main()

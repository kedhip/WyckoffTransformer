from typing import Dict, Iterable, Set, FrozenSet, Optional, List, Tuple
from itertools import chain
from operator import attrgetter, itemgetter
from functools import partial
from collections import defaultdict
from enum import Enum
from pathlib import Path
import gzip
import pickle
from pandas import DataFrame
import argparse
import torch
import omegaconf

ServiceToken = Enum('ServiceToken', ['PAD', 'STOP', 'MASK'])

class EnumeratingTokeniser(dict):
    @classmethod
    def from_token_set(cls,
        all_tokens: Set|FrozenSet,
        max_tokens: Optional[int] = None):
        for special_token in ServiceToken:
            if special_token.name in all_tokens:
                raise ValueError(f"Special token {special_token.name} is in the dataset")
        instance = cls()
        instance.update({token: idx for idx, token in enumerate(
            chain(all_tokens, map(attrgetter('name'), ServiceToken)))})
        instance.stop_token = instance[ServiceToken.STOP.name]
        instance.pad_token = instance[ServiceToken.PAD.name]
        instance.mask_token = instance[ServiceToken.MASK.name]
        # Theoretically, we can check it in the beginnig, but
        # the performance hit is negligible
        if max_tokens is not None and len(instance) > max_tokens:
            raise ValueError(f"Too many tokens: {len(instance)}. Remember "
            f"that we also added {len(ServiceToken)} service tokens")
        return instance


    def tokenise_sequence(self,
                          sequence: Iterable,
                          original_max_len: int,
                          **tensor_args) -> torch.Tensor:
        tokenised_sequence = [self[token] for token in sequence]
        padding = [self.pad_token] * (original_max_len - len(tokenised_sequence))
        return torch.tensor(tokenised_sequence + [self.stop_token] + padding, **tensor_args)
    

    def tokenise_single(self, token, **tensor_args) -> torch.Tensor:
        return torch.tensor(self[token], **tensor_args)


def tokenise_dataset(datasets_pd: Dict[str, DataFrame],
                     config: omegaconf.OmegaConf) -> \
                        Tuple[Dict[str, Dict[str, torch.Tensor|List[List[torch.Tensor]]]], Dict[str, EnumeratingTokeniser]]:
    tokenisers = {}
    dtype = getattr(torch, config.dtype)
    max_tokens = torch.iinfo(dtype).max
    for token_field in config.token_fields.pure_categorical:
        all_tokens = frozenset(chain.from_iterable(chain.from_iterable(map(itemgetter(token_field), datasets_pd.values()))))
        tokenisers[token_field] = EnumeratingTokeniser.from_token_set(all_tokens, max_tokens)

    for sequence_field in config.sequence_fields.pure_categorical:
        all_tokens = frozenset(chain.from_iterable(map(lambda df: df[sequence_field].unique(), datasets_pd.values())))
        tokenisers[sequence_field] = EnumeratingTokeniser.from_token_set(all_tokens, max_tokens)

    # We don't check consistency among the fields here
    # The value is for the original sequences, withot service tokens
    original_max_len = max(map(len, chain.from_iterable(
        map(itemgetter(config.token_fields.pure_categorical[0]),
            datasets_pd.values()))))
    
    tensors = defaultdict(dict)
    for dataset_name, dataset in datasets_pd.items():
        for field in config.token_fields.pure_categorical:
            tensors[dataset_name][field] = torch.stack(
                dataset[field].map(partial(
                    tokenisers[field].tokenise_sequence,
                    original_max_len=original_max_len,
                    dtype=dtype)).to_list())
        for field in config.sequence_fields.pure_categorical:
            tensors[dataset_name][field] = torch.stack(
                dataset[field].map(partial(
                    tokenisers[field].tokenise_single,
                    original_max_len=original_max_len,
                    dtype=dtype)).to_list())
        for field in config.augmented_token_fields:
            tensors[dataset_name][field] = dataset[field].map(lambda variants:
                    [tokenisers[field].tokenise(
                        variant, original_max_len=original_max_len, dtype=dtype)
                        for variant in variants]).to_list()
    # +1 as we have added the stop token
    return tensors, tokenisers


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

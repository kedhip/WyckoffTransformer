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
                    dtype=dtype)).to_list())
        for field in config.augmented_token_fields:
            augmented_field = f"{field}_augmented"
            tensors[dataset_name][augmented_field] = dataset[augmented_field].map(lambda variants:
                    [tokenisers[field].tokenise_sequence(
                        variant, original_max_len=original_max_len, dtype=dtype)
                        for variant in variants]).to_list()
    return tensors, tokenisers


def load_tensors_and_tokenisers(
    dataset: str,
    config_name: str,
    cache_path: Path = Path(__file__).parent.parent.resolve() / "cache"):
    
    this_cache_path = cache_path / dataset
    with gzip.open(this_cache_path / 'tensors' / f'{config_name}.pkl.gz', "rb") as f:
        tensors = pickle.load(f)
    with gzip.open(this_cache_path / 'tokenisers' / f'{config_name}.pkl.gz', "rb") as f:
        tokenisers = pickle.load(f)
    return tensors, tokenisers

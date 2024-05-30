from typing import List, Tuple
import torch
from torch import Tensor


def randint_tensor(high: torch.Tensor) -> Tensor:
    """
    Generates a random tensor of integers from 0 (inclusive) to high (exclusive).
    
    The probability distribution is not perfectly uniform in the case of
    torch.iinfo(dtype).max - torch.iinfo(dtype).min % high != 0
    but it's good enough for our purposes, where high < 50, and max-min ~ 2^64
    
    Arguments:
        high: The maximum value of the random tensor.
    Returns:
        A random tensor of integers from 0 to high - 1, size high.size.
    """
    dtype = high.dtype
    return torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, size=high.size(),
        device=high.device, dtype=dtype) % high


class AugmentedCascadeDataset():
    def __init__(
        self,
        data: dict[Tensor|List[Tensor]],
        cascade_order: Tuple[str, ...],
        masks: dict[str, int|Tensor],
        pads: dict[str, int|Tensor],
        start_field: str,
        augmented_field: str,
        dtype: torch.dtype = torch.int64,
        device: str = "cpu"):
        """
        A class for contaning augmented cascade datasets.
        Cascade means that a dataset is a list of tensors, where each tensor is a different field, with
        each subsequent field depending on the previous one. For use in the Wyckoff transformer. Augmented
        means that for up to one field we have multiple possible values, and we want to sample from them.

        Arguments:
            data: A dictionary of tensors (for normal fields) or lists of tensors (for augmented fields),
                with keys being the field names.
            cascade_order: A tuple of field names, in the order they should be passed to the model.
            masks: A dictionary of mask values for each field
            start_field: The name of the field that should be used as the start token.
            augmented_field: The name of the field that should be augmented.
            dtype: The dtype of the tensors.
            device: The device of the tensors.
        """
        self.cascade_order = cascade_order
        self.cascade_index_from_field = {name: i for i, name in enumerate(cascade_order)}
        self.augmented_field = augmented_field
        self.augmented_field_index = cascade_order.index(augmented_field)
        self.masks = {name: torch.tensor(masks[name], dtype=dtype, device=device) for name in cascade_order}
        self.pads = {name: torch.tensor(pads[name], dtype=dtype, device=device) for name in cascade_order}

        self.start_tokens = data[start_field].type(dtype).to(device)
        self.data = {name: data[name].type(dtype).to(device) for name in cascade_order if name != augmented_field}
        
        # Ideally, we would like a nested tensor, but it doesn't support indexing we need
        # So we use a custom data store, and indices to it
        self.sequence_length = self.data[self.cascade_order[0]].size(1)
        self.augmentation_variants = torch.tensor(
            list(map(len, data[f"{augmented_field}_augmented"])), dtype=dtype, device=device)
        self.augmentation_start_indices = (torch.cumsum(self.augmentation_variants, dim=0) -
            self.augmentation_variants) * self.sequence_length
        # It will be used in torch.gather
        self.augmentation_data_store = torch.cat([torch.cat(x, dim=0) for x in data[f"{augmented_field}_augmented"]],
            dim=0).type(dtype).to(device).unsqueeze(0).expand(self.augmentation_variants.size(0), -1)


    def __len__(self):
        return self.data[self.cascade_order[0]].size(0)


    # Compilation is safe since the function only ever uses the same data
    @torch.compile
    def get_augmentation(self):
        augnementation_selection = randint_tensor(self.augmentation_variants)
        selection_start = augnementation_selection*self.sequence_length + self.augmentation_start_indices
        selection_indices = selection_start.unsqueeze(1) + torch.arange(
            self.sequence_length, device=selection_start.device, dtype=selection_start.dtype)
        return torch.gather(self.augmentation_data_store, 1, selection_indices)


    def get_masked_cascade_data(
        self,
        known_seq_len: int,
        known_cascade_len: int):
            # assert known_seq_len < data.shape[1]
            # assert known_seq_len >= 0

            augmented_data = self.get_augmentation()
            if known_cascade_len == self.augmented_field_index:
                target = augmented_data[:, known_seq_len]
            else:
                target = self.data[self.cascade_order[known_cascade_len]][:, known_seq_len]
            target_is_viable = (target != self.pads[self.cascade_order[known_cascade_len]])
            
            # We are predicting the first cascade element also through the mask
            # It would be slightly more efficient to predict it using only the previous data
            res = []
            for cascade_index, name in enumerate(self.cascade_order):
                if name == self.augmented_field:
                    cascade_vector = augmented_data[target_is_viable]
                else:
                    cascade_vector = self.data[name][target_is_viable]
                if cascade_index < known_cascade_len:
                    res.append(cascade_vector[:, :known_seq_len + 1])
                else:
                    res.append(torch.cat([
                        cascade_vector[:, :known_seq_len],
                        self.masks[name].expand(cascade_vector.size(0), 1)], dim=1))
            return self.start_tokens[target_is_viable], res, target[target_is_viable]
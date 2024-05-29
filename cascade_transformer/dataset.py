import torch


def randint_tensor(high: torch.Tensor) -> Tensor:
    """
    Generates a random tensor of integers from 0 to high - 1, size high.size.
    """
    dtype = high.dtype
    # The probability distribution is not perfect, as max-min won't always be divisible by high,
    # but it's good enough for our purposes, where high < 50, and max-min ~ 2^64
    return torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, size=high.size(),
        device=high.device, dtype=dtype) % high


class AugmentedCascadeDataset():
    def __init__(
        self,
        data: dict[Tensor|List[Tensor]],
        cascade_order: Tuple[str, ...],
        start_field: str,
        augmented_field: str,
        dtype: torch.dtype = torch.int64,
        device: str = "cpu"):
    self.start_tokes = data[start_field].tpype(dtype).to(device)
    self.data = data

    self.augmented_field = augmented_field
    self.augmentation_variants = torch.tensor(
        list(map(len, data[f"{augmented_field}_augmented"])), dtype=dtype, device=device)
    self.augmentation_tensor = torch.nested.nested_tensor(
        data[f"{augmented_field}_augmented"], dtype=dtype, device=device).to_padded_tensor(
        padding=pad[augmented_field])

    #self.train_datasets = {name: torch_datasets["train"][name].type(dtype).to(device) for name in cascade_order if name != augmented_field}
    #self.train_start = torch_datasets["train"][start_name].type(dtype).to(device)
    
    def get_augmented_cascade(self):
        augnementation_selection = randint_tensor(self.augmentation_variants)
        augmentated_data = self.augmentation_tensor[
            torch.arange(self.augmentation_tensor.size(0), device=self.augmentation_tensor.device),
            augnementation_selection]
    
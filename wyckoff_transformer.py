from typing import Tuple
from random import randint
import pathlib
import datetime
import torch
from torch import nn
from torch import Tensor
from cascade_transformer.model import get_cascade_target, get_masked_cascade_data
import wandb


def randint_tensor(high: torch.Tensor):
    dtype = high.dtype
    return torch.randint(torch.iinfo(dtype).max, size=(len(high),), device=high.device) % high


class WyckoffGenerator():
    def __init__(self,
                 model: nn.Module,
                 cascade_order: Tuple,
                 mask: dict,
                 max_len: int,
                 device: str,
                 dtype = torch.int64):
        self.model = model
        self.max_len = max_len
        self.cascade_order = cascade_order
        self.device = device
        self.mask_tensor = torch.tensor([mask[name] for name in cascade_order], dtype=dtype, device=device)


    def generate_tensors(self, start: Tensor) -> Tensor:
        """
        Generates a sequence of tokens.

        Arguments:
            start: The start token. It should be a tensor of shape [batch_size, 1, len(cascade_order)].

        Returns:
            The generated sequence of tokens. It has shape [batch_size, max_len, len(cascade_order)].
        """
        self.model.eval()
        with torch.no_grad():
            batch_size = start.size(0)
            generated = [torch.empty(batch_size, self.max_len, dtype=start.dtype, device=self.device) \
                for _ in range(len(self.cascade_order))]
            for known_seq_len in range(self.max_len):
                for known_cascade_len in range(len(self.cascade_order)):
                    masked_data = get_masked_cascade_data(
                        generated, self.mask_tensor, known_seq_len, known_cascade_len)
                    model_output = self.model(start, masked_data, None, known_cascade_len)
                    probas = torch.nn.functional.softmax(model_output, dim=1)
                    # +1 as START
                    generated[known_cascade_len][:, known_seq_len] = torch.multinomial(probas, num_samples=1).squeeze()
            return generated

class WyckoffTrainer():
    def __init__(self,
                 model: nn.Module,
                 torch_datasets: dict,
                 pad: dict,
                 mask: dict,
                 cascade_order: Tuple,
                 augmented_field: str|None,
                 start_name: str,
                 max_len: int,
                 device: str,
                 initial_lr: float = 1.,
                 clip_grad_norm: float = 0.5,
                 # batch_size: int = 1024*32,
                 val_period: int = 10,
                 patience: int = 10,
                 dtype = torch.int64):
        """
        Arguments:
        val_period: How often to evaluate the model on the validation dataset.
        patience: How many validations to wait before reducing the learning rate.
        dtype: We actually want dtype uint8. However, Embedding layer does not support it,
            and CrossEntropyLoss expects only int64.
        """
        # Sequences have difference lengths, so we need to make sure that
        # long sequences don't dominate the loss
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.9, patience=patience)
        self.max_len = max_len
        self.model = model
        self.dtype = dtype
        self.pad_tensor = torch.tensor([pad[name] for name in cascade_order], dtype=self.dtype, device=device)
        self.mask_tensor = torch.tensor([mask[name] for name in cascade_order], dtype=self.dtype, device=device)
        self.train_datasets = {name: torch_datasets["train"][name].type(dtype).to(device) for name in cascade_order if name != augmented_field}
        self.train_start = torch_datasets["train"][start_name].type(dtype).to(device)
        self.train_augmentation_variants = torch.tensor(
            list(map(len, torch_datasets["train"][f"{augmented_field}_augmented"])), dtype=dtype, device=device)
        self.train_augmentation_tensor = torch.nested.nested_tensor(
            torch_datasets["train"][f"{augmented_field}_augmented"], dtype=dtype, device=device).to_padded_tensor(
            padding=pad[augmented_field])
        self.augmented_field = augmented_field
        
        # self.train_loader = torch.utils.data.DataLoader(
            # torch.utils.data.TensorDataset(self.torch_datasets["train"][start_name].type(dtype).to(device), train_dataset),
            # batch_size=batch_size,
            # With the new code we still can process the whole dataset
            # shuffle=False)
        self.val_start = torch_datasets["val"][start_name].type(dtype).to(device)
        self.val_datasets = {name: torch_datasets["val"][name].type(dtype).to(device) for name in cascade_order if name != augmented_field}
        self.val_augmentation_variants = torch.tensor(
            list(map(len, torch_datasets["val"][f"{augmented_field}_augmented"])), dtype=dtype, device=device)
        self.val_augmentation_tensor = torch.nested.nested_tensor(
            torch_datasets["val"][f"{augmented_field}_augmented"], dtype=dtype, device=device).to_padded_tensor(
            padding=pad[augmented_field])

        self.cascade_len = len(cascade_order)
        self.cascade_order = cascade_order
        self.patience = patience
        self.val_period = val_period
        self.clip_grad_norm = clip_grad_norm
        self.config = {
            # "batch_size": self.train_loader.batch_size,
            "cascade_order": cascade_order,
            "use_mixer": model.use_mixer,
            "n_head": model.n_head,
            "n_layers": model.n_layers,
            "d_hid": model.d_hid,
            "dropout": model.dropout,
            "initial_lr": initial_lr,
            "clip_grad_norm": clip_grad_norm,
            "max_len": self.max_len}


    def get_loss(self,
        start_batch: Tensor,
        data_batch: Tensor,
        known_seq_len: int,
        known_cascade_len: int) -> Tensor:
        target = get_cascade_target(data_batch, known_seq_len, known_cascade_len)
        non_pad_target = (target != self.pad_tensor[known_cascade_len])
        selected_target = target[non_pad_target]
        #if len(selected_target) == 0:
        #    return torch.zeros(1)
        #else:
        masked_data = get_masked_cascade_data(
            [data[non_pad_target] for data in data_batch], self.mask_tensor, known_seq_len, known_cascade_len)
        # No padding, as we have already discarded the padding
        prediction = self.model(start_batch[non_pad_target], masked_data, None, known_cascade_len)
        return self.criterion(prediction, selected_target)


    def train_epoch(self):
        self.model.train()
        #for start_batch, data_batch in self.train_loader:
        self.optimizer.zero_grad(set_to_none=True)
        # TODO is STOP in max_len?
        known_seq_len = randint(0, self.max_len)
        known_cascade_len = randint(0, self.cascade_len - 1)
        augnementation_selection = randint_tensor(self.train_augmentation_variants)
        augmentated_data = self.train_augmentation_tensor[
            torch.arange(self.train_augmentation_tensor.size(0), device=self.train_augmentation_tensor.device),
            augnementation_selection]
        loss = self.get_loss(
            self.train_start,
            [self.train_datasets[name] if name != self.augmented_field else augmentated_data for name in self.cascade_order],
            known_seq_len, known_cascade_len)
        #if loss != 0:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        wandb.log({"train_loss_batch": loss,
                   "known_seq_len": known_seq_len,
                   "known_cascade_len": known_cascade_len})


    def evaluate(self) -> Tensor:
        """
        Evaluates the model by calculating the average loss on the validation dataset.

        Returns:
            The average loss on the validation dataset.
        """
        self.model.eval()
        val_loss = 0.
        train_loss = 0.
        with torch.no_grad():
            for known_seq_len in range(0, self.max_len):
                for known_cascade_len in range(0, self.cascade_len):
                    augnementation_selection = randint_tensor(self.train_augmentation_variants)
                    augmentated_data = self.train_augmentation_tensor[
                        torch.arange(self.train_augmentation_tensor.size(0), device=self.train_augmentation_tensor.device),
                        augnementation_selection]
                    train_loss += self.get_loss(
                        self.train_start,
                        [self.train_datasets[name] if name != self.augmented_field else augmentated_data for name in self.cascade_order],
                        known_seq_len, known_cascade_len)
                    
                    augnementation_selection = randint_tensor(self.val_augmentation_variants)
                    augmentated_data = self.val_augmentation_tensor[
                        torch.arange(self.val_augmentation_tensor.size(0), device=self.val_augmentation_tensor.device),
                        augnementation_selection]
                    val_loss += self.get_loss(
                        self.val_start,
                        [self.val_datasets[name] if name != self.augmented_field else augmentated_data for name in self.cascade_order],
                        known_seq_len, known_cascade_len)
            # ln(P) = ln p(t_n|t_n-1, ..., t_1) + ... + ln p(t_2|t_1)
            # We are minimising the negative log likelihood of the whole sequences
            return (val_loss / self.val_start.shape[0]).item(), \
                   (train_loss / self.train_start.shape[0]).item()


    def train(self, epochs: int):
        best_val_loss = float('inf')
        run_path = pathlib.Path("checkpoints", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        run_path.mkdir(parents=True)
        best_model_params_path = run_path / "best_model_params.pt"

        with wandb.init(
            project="WyckoffTransformer",
            job_type="train",
            config=self.config):
            wandb.config.update({"epochs": epochs})
            for epoch in range(epochs):
                self.train_epoch()
                if epoch % self.val_period == 0:
                    val_loss_epoch, train_loss_eposh = self.evaluate()
                    wandb.log({"val_loss_epoch": val_loss_epoch,
                               "train_loss_epoch": train_loss_eposh,
                               "lr": self.optimizer.param_groups[0]['lr'],
                               "epoch": epoch}, commit=False)
                    if val_loss_epoch < best_val_loss:
                        best_val_loss = val_loss_epoch
                        torch.save(self.model.state_dict(), best_model_params_path)
                        wandb.save(str(best_model_params_path))
                        print(f"Epoch {epoch} val_loss_epoch {val_loss_epoch} saved to {best_model_params_path}")
                    self.scheduler.step(val_loss_epoch)
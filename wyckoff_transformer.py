from typing import Tuple
from random import randint
import pathlib
import datetime
import torch
from torch import nn
from torch import Tensor
from cascade_transformer.dataset import AugmentedCascadeDataset
import wandb


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
        self.device = device
        self.pad_tensor = torch.tensor([pad[name] for name in cascade_order], dtype=self.dtype, device=device)
        self.mask_tensor = torch.tensor([mask[name] for name in cascade_order], dtype=self.dtype, device=device)
        self.augmented_field = augmented_field
        self.train_dataset = AugmentedCascadeDataset(
            data=torch_datasets["train"],
            cascade_order=cascade_order,
            masks=mask,
            pads=pad,
            start_field=start_name,
            augmented_field=augmented_field,
            dtype=dtype,
            device=device)
        
        self.val_dataset = AugmentedCascadeDataset(
            data=torch_datasets["val"],
            cascade_order=cascade_order,
            masks=mask,
            pads=pad,
            start_field=start_name,
            augmented_field=augmented_field,
            dtype=dtype,
            device=device)
    
        self.cascade_len = len(cascade_order)
        self.cascade_order = cascade_order
        self.patience = patience
        self.val_period = val_period
        self.clip_grad_norm = clip_grad_norm
        self.config = {
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
        dataset: AugmentedCascadeDataset,
        known_seq_len: int,
        known_cascade_len: int) -> Tensor:
        start_tokens, masked_data, target = dataset.get_masked_cascade_data(known_seq_len, known_cascade_len)
        # No padding, as we have already discarded the padding
        prediction = self.model(start_tokens, masked_data, None, known_cascade_len)
        return self.criterion(prediction, target)


    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        known_seq_len = randint(0, self.max_len)
        known_cascade_len = randint(0, self.cascade_len - 1)
        loss = self.get_loss(self.train_dataset, known_seq_len, known_cascade_len)
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
        val_loss = torch.zeros(self.cascade_len, device=self.device)
        train_loss = torch.zeros(self.cascade_len, device=self.device)
        with torch.no_grad():
            for known_seq_len in range(0, self.max_len):
                for known_cascade_len in range(0, self.cascade_len):
                    train_loss[known_cascade_len] += self.get_loss(
                        self.train_dataset, known_seq_len, known_cascade_len)
                    val_loss[known_cascade_len] += self.get_loss(
                        self.val_dataset, known_seq_len, known_cascade_len)
            # ln(P) = ln p(t_n|t_n-1, ..., t_1) + ... + ln p(t_2|t_1)
            # We are minimising the negative log likelihood of the whole sequences
            return val_loss / len(self.val_dataset), \
                   train_loss / len(self.train_dataset)


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
                    val_loss_epoch, train_loss_epoch = self.evaluate()
                    val_loss_dict = {name: val_loss_epoch[i].item() for i, name in enumerate(self.cascade_order)}
                    total_val_loss = val_loss_epoch.sum()
                    val_loss_dict["total"] = total_val_loss.item()
                    train_loss_dict = {name: train_loss_epoch[i].item() for i, name in enumerate(self.cascade_order)}
                    train_loss_dict["total"] = train_loss_epoch.sum().item()
                    wandb.log({"val_loss_epoch": val_loss_dict,
                                "train_loss_epoch": train_loss_dict,
                               "lr": self.optimizer.param_groups[0]['lr'],
                               "epoch": epoch}, commit=False)
                    if total_val_loss < best_val_loss:
                        best_val_loss = total_val_loss
                        torch.save(self.model.state_dict(), best_model_params_path)
                        wandb.save(str(best_model_params_path))
                        print(f"Epoch {epoch} val_loss_epoch {total_val_loss} saved to {best_model_params_path}")
                    self.scheduler.step(total_val_loss)
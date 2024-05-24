from typing import Tuple
from random import randint
import pathlib
import datetime
import torch
from torch import nn
from torch import Tensor
from cascade_transformer.model import get_cascade_target, get_masked_cascade_data
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
            batch_size = start.shape[0]
            generated = torch.empty(batch_size, self.max_len, len(self.cascade_order), dtype=start.dtype, device=self.device)
            for known_seq_len in range(self.max_len):
                for known_cascade_len in range(len(self.cascade_order)):
                    masked_data = get_masked_cascade_data(
                        generated, self.mask_tensor, known_seq_len, known_cascade_len)
                    model_output = self.model(start, masked_data, None, known_cascade_len)
                    probas = torch.nn.functional.softmax(model_output, dim=1)
                    # +1 as START
                    generated[:, known_seq_len, known_cascade_len] = torch.multinomial(probas, num_samples=1).squeeze()
            return generated

class WyckoffTrainer():
    def __init__(self,
                 model: nn.Module,
                 torch_datasets: dict,
                 pad: dict,
                 mask: dict,
                 cascade_order: Tuple,
                 start_name: str,
                 max_len: int,
                 device: str,
                 batch_size: int = 1024*32,
                 dtype = torch.int64):
        """
        We actually want dtype uint8. However, Embedding layer does not support it,
        and CrossEntropyLoss expects only int64.
        """
        # Sequences have difference lengths, so we need to make sure that
        # long sequences don't dominate the loss
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.lr = 5.0
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.9, patience=10)
        self.max_len = max_len
        self.model = model
        self.torch_datasets = torch_datasets
        self.dtype = dtype
        self.pad_tensor = torch.tensor([pad[name] for name in cascade_order], dtype=self.dtype, device=device)
        self.mask_tensor = torch.tensor([mask[name] for name in cascade_order], dtype=self.dtype, device=device)
        train_dataset = torch.stack([self.torch_datasets["train"][name] for name in cascade_order], dim=2).type(dtype).to(device)
        # TODO shuffle ?
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.torch_datasets["train"][start_name].type(dtype).to(device), train_dataset),
            batch_size=batch_size,
            # With the new code we still can process the whole dataset
            shuffle=False)
        self.val_start = self.torch_datasets["val"][start_name].type(dtype).to(device)
        self.val_dataset = torch.stack([self.torch_datasets["val"][name] for name in cascade_order], dim=2).type(dtype).to(device)
        self.cascade_len = len(cascade_order)
        self.config = {
            "batch_size": self.train_loader.batch_size,
            "cascade_order": cascade_order,
            "use_mixer": model.use_mixer,
            "max_len": self.max_len}


    def get_loss(self,
        start_batch: Tensor,
        data_batch: Tensor,
        known_seq_len,
        known_cascade_len) -> Tensor:
        target = get_cascade_target(data_batch, known_seq_len, known_cascade_len)
        non_pad_target = (target != self.pad_tensor[known_cascade_len])
        selected_target = target[non_pad_target]
        if len(selected_target) == 0:
            return torch.zeros(1)
        else:
            masked_data = get_masked_cascade_data(data_batch[non_pad_target], self.mask_tensor, known_seq_len, known_cascade_len)
            # No padding, as we have already discarded the padding
            prediction = self.model(start_batch[non_pad_target], masked_data, None, known_cascade_len)
            return self.criterion(prediction, selected_target)


    def train_epoch(self):
        self.model.train()
        for start_batch, data_batch in self.train_loader:
            # TODO is STOP in max_len?
            known_seq_len = randint(0, self.max_len)
            known_cascade_len = randint(0, self.cascade_len - 1)
            loss = self.get_loss(start_batch, data_batch, known_seq_len, known_cascade_len)
            if loss != 0:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.lr = self.scheduler.get_last_lr()[0]
                wandb.log({"train_loss_batch": loss,
                           "known_seq_len": known_seq_len,
                           "known_cascade_len": known_cascade_len,
                           "lr": self.lr})


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
                    val_loss += self.get_loss(self.val_start, self.val_dataset, known_seq_len, known_cascade_len)
                    train_loss += self.get_loss(self.train_loader.dataset.tensors[0], self.train_loader.dataset.tensors[1], known_seq_len, known_cascade_len)
            # ln(P) = ln p(t_n|t_n-1, ..., t_1) + ... + ln p(t_2|t_1)
            # We are minimising the negative log likelihood of the whole sequences
            return (val_loss / self.torch_datasets["val"]["symmetry_sites"].shape[0]).item(), \
                   train_loss.item() / len(self.train_loader.dataset)


    def train(self, epochs: int, val_period: int = 10):
        best_val_loss = float('inf')
        run_path = pathlib.Path("checkpoints", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        run_path.mkdir(parents=True)
        best_model_params_path = run_path / "best_model_params.pt"

        with wandb.init(
            project="WyckoffTransformer",
            job_type="train",
            config=self.config):
            wandb.config.update({
                "epochs": epochs,
                "val_period": val_period})
            for epoch in range(1, epochs + 1):
                self.train_epoch()
                if epoch % val_period == 0:
                    val_loss_epoch, train_loss_eposh = self.evaluate()
                    wandb.log({"val_loss_epoch": val_loss_epoch,
                               "train_loss_epoch": train_loss_eposh,
                               "epoch": epoch}, commit=False)
                    if val_loss_epoch < best_val_loss:
                        best_val_loss = val_loss_epoch
                        wandb.save(str(best_model_params_path))
                        torch.save(self.model.state_dict(), best_model_params_path)
                        print(f"Epoch {epoch} val_loss_epoch {val_loss_epoch} saved to {best_model_params_path}")
                    self.scheduler.step(val_loss_epoch, epoch=epoch)
from typing import Tuple
from random import randint
import pathlib
import datetime
import torch
from torch import nn
from torch import Tensor
import wandb
from omegaconf import OmegaConf

from cascade_transformer.dataset import AugmentedCascadeDataset
from cascade_transformer.model import CascadeTransformer
from wyckoff_transformer.tokenization import load_tensors_and_tokenisers


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
                 tokenisers: dict,
                 cascade_order: Tuple,
                 augmented_field: str|None,
                 start_name: str,
                 device: torch.DeviceObjType,
                 optimisation_config: dict):
        # Nothing else will work in foreseeable future
        self.dtype = torch.int64
        # Sequences have difference lengths, so we need to make sure that
        # long sequences don't dominate the loss
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.optimizer = getattr(torch.optim, optimisation_config.optimiser.name)(
            model.parameters(), **optimisation_config.optimiser.config)
        self.scheduler = getattr(torch.optim.lr_scheduler, optimisation_config.scheduler.name)(
            self.optimizer, 'min', **optimisation_config.scheduler.config)
        
        self.model = model
        self.device = device
        self.augmented_field = augmented_field

        masks_dict = {field: tokenisers[field].mask_token for field in cascade_order}
        pad_dict = {field: tokenisers[field].pad_token for field in cascade_order}

        self.train_dataset = AugmentedCascadeDataset(
            data=torch_datasets["train"],
            cascade_order=cascade_order,
            masks=masks_dict,
            pads=pad_dict,
            start_field=start_name,
            augmented_field=augmented_field,
            dtype=self.dtype,
            device=self.device)
        
        self.val_dataset = AugmentedCascadeDataset(
            data=torch_datasets["val"],
            cascade_order=cascade_order,
            masks=masks_dict,
            pads=pad_dict,
            start_field=start_name,
            augmented_field=augmented_field,
            dtype=self.dtype,
            device=device)
        assert self.train_dataset.sequence_length == self.val_dataset.sequence_length
    
        self.clip_grad_norm = optimisation_config.clip_grad_norm
        self.cascade_len = len(cascade_order)
        self.cascade_order = cascade_order
        self.epochs = optimisation_config.epochs
        self.validation_period = optimisation_config.validation_period
        self.early_stopping_patience_epochs = optimisation_config.early_stopping_patience_epochs


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
        known_seq_len = randint(0, self.train_dataset.sequence_length - 1)
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
            for known_seq_len in range(0, self.train_dataset.sequence_length):
                for known_cascade_len in range(0, self.cascade_len):
                    train_loss[known_cascade_len] += self.get_loss(
                        self.train_dataset, known_seq_len, known_cascade_len)
                    val_loss[known_cascade_len] += self.get_loss(
                        self.val_dataset, known_seq_len, known_cascade_len)
            # ln(P) = ln p(t_n|t_n-1, ..., t_1) + ... + ln p(t_2|t_1)
            # We are minimising the negative log likelihood of the whole sequences
            return val_loss / len(self.val_dataset), \
                   train_loss / len(self.train_dataset)


    def train(self):
        best_val_loss = float('inf')
        best_val_epoch = 0
        run_path = pathlib.Path("checkpoints", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        run_path.mkdir(parents=True)
        best_model_params_path = run_path / "best_model_params.pt"
        
        for epoch in range(self.epochs):
            self.train_epoch()
            if epoch % self.validation_period == 0:
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
                    best_val_epoch = epoch
                    torch.save(self.model.state_dict(), best_model_params_path)
                    wandb.save(str(best_model_params_path))
                    print(f"Epoch {epoch} val_loss_epoch {total_val_loss} saved to {best_model_params_path}")
                    wandb.log({"best_val_loss": best_val_loss}, commit=False)
                if epoch - best_val_epoch > self.early_stopping_patience_epochs:
                    print(f"Early stopping at epoch {epoch} after more than {self.early_stopping_patience_epochs}"
                           " epochs without improvement")
                    return
                self.scheduler.step(total_val_loss)


def train_from_config(config_dict: dict, device: torch.device):
    config = OmegaConf.create(config_dict)
    if len(config.tokeniser.augmented_token_fields) > 1:
        raise ValueError("Only one augmented field is supported")

    tensors, tokenisers = load_tensors_and_tokenisers(config.dataset, config.tokeniser.name)

    if "cascade_order" in config.model:
        cascade_order = config.model.cascade_order
        print(f"Using cascade order {cascade_order}")
    else:
        cascade_order = tuple(config.model.cascade_embedding_size.keys())

    full_cascade = dict()
    for field in cascade_order:
        full_cascade[field] = (len(tokenisers[field]),
                               config.model.cascade_embedding_size[field],
                               tokenisers[field].pad_token)

    model = CascadeTransformer(
        n_start=len(tokenisers[config.model.start_token]),
        cascade=full_cascade.values(),
        **config.model.CascadeTransformer_args
        ).to(device)


    trainer = WyckoffTrainer(
        model, tensors, tokenisers, tuple(full_cascade.keys()), 
        config.tokeniser.augmented_token_fields[0],
        config.model.start_token, device, optimisation_config=config.optimisation)
    trainer.train()

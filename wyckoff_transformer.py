from typing import Tuple
from random import randint
import math
import pathlib
import datetime
import os
import torch
from torch import nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import wandb

class WyckoffTransformerModel(nn.Module):

    def __init__(self, n_space_groups: int, n_sites: int, n_elements: int, d_space_groups: int, d_sites: int, d_species: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        # We are going to concatenate the site, the element, and the space group
        self.d_model = d_sites + d_species + d_space_groups
        self.n_token = n_sites + n_elements
        self.n_elements = n_elements
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.sites_embedding = nn.Embedding(n_sites, d_sites)
        self.species_embedding = nn.Embedding(n_elements, d_species)
        self.space_groups_embedding = nn.Embedding(n_space_groups, d_space_groups)
        # We predict both the site and the element
        # However! They can not be sampled independently!
        # We are going to condition site on the element,
        # so that we can attempt CSP
        # The question of autoregressive construction order is still open though
        # An option would be to first sort by element, and then by site multiplicity
        self.linear = nn.Linear(self.d_model, self.n_token)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # CONSIDER(kazeevn) using the site symmetry matrices to initialize the site embeddings
        # and some inforamtion about the element to initialize the element embeddings
        self.sites_embedding.weight.data.uniform_(-initrange, initrange)
        self.species_embedding.weight.data.uniform_(-initrange, initrange)
        self.space_groups_embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self,
                space_group: Tensor,
                sites: Tensor,
                species: Tensor,
                padding_mask: Tensor) -> (Tensor, Tensor):
        """
        Arguments:
            space_group: Tensor of shape ``[batch_size]``
            sites: Tensor, shape ``[seq_len, batch_size]``
            species: Tensor, shape ``[seq_len, batch_size]``

        Returns:
            Tensor of shape ``[seq_len, batch_size, n_elements]``
            Tensor of shape ``[seq_len, batch_size, n_sites]``
            Remember not to train via independent sampling!
        """
        # And here is a good question on masking
        # 1. Padding. If we just mask it, the model won't be able to stop predicting, we'll need to additinally introduce a stop token.
        # probably, can do without for now. The previous iteration reliably predicted sequences of PADs
        # 2. Casual masking. We can't use it as is, as we need the model to attend on the element of the site
        # 3. While we output the whole sequence, we should only train on the last element, as we use MASK for marking the site under prediction
        # and since we do incremental batching, using the previous elements multiple time would unjustly increase their weight
        src = torch.concat([self.sites_embedding(sites), self.species_embedding(species),
                            self.space_groups_embedding(torch.tile(space_group, (sites.shape[0], 1)))], dim=2) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)
        output = self.linear(output)
        return output[:, :, :self.n_elements], output[:, :, self.n_elements:]


def get_batches(
    symmetry_sites: Tensor,
    symmetry_elements:Tensor,
    element_seq_len: int,
    mask_id_tensor: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Args:
        element_seq_len: number of elements considered known
    Returns:
        Two tuples of (data, target, padding_mask):
        1. The first is to predict the element given i sites and elements
        2. The second is to predict the site given i sites and i+1 elements
        
    """
    # Can't predict past the last token
    assert element_seq_len < symmetry_elements.shape[0]
    # Predict element number element_seq_len + 1
    data_element = {
        "sites": symmetry_sites[0:element_seq_len],
        "species": symmetry_elements[0:element_seq_len]
    }
    target_element = symmetry_elements[element_seq_len]

    # Predict site number element_seq_len + 1
    # For sites, we condition on species, hence the + 1  in species
    data_site = {
        "sites": torch.cat([symmetry_sites[0:element_seq_len], mask_id_tensor.expand(1, symmetry_sites.shape[1])], dim=0),
        "species": symmetry_elements[0:element_seq_len + 1]
    }
    target_site = symmetry_sites[element_seq_len]
    return ((data_element, target_element), (data_site, target_site))


class WyckoffTrainer():
    def __init__(self, model: nn.Module, torch_datasets: dict, max_len: int, mask_id_tensor: Tensor):
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 5.0  # learning rate
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 20, gamma=0.95)
        self.max_len = max_len
        self.model = model
        self.torch_datasets = torch_datasets
        self.mask_id_tensor = mask_id_tensor

    def get_loss(self, model, dataset, known_seq_len):
        (data_element, target_element), (data_site, target_site) = get_batches(dataset["symmetry_sites"],
                                                                               dataset["symmetry_elements"],
                                                                               known_seq_len,
                                                                               self.mask_id_tensor)
        output_element = model(dataset["spacegroup_number"], **data_element,
                            padding_mask=dataset["padding_mask"][:, :known_seq_len])[0]
        loss_element = self.criterion(output_element[-1, :, :], target_element)
        output_site = model(dataset["spacegroup_number"], **data_site,
                            padding_mask=dataset["padding_mask"][:, :known_seq_len + 1])[1]
        loss_site = self.criterion(output_site[-1, :, :], target_site)
        loss = loss_element + loss_site
        return loss


    def train_step(self):
        self.model.train()
        known_seq_len = randint(1, self.max_len - 1)
        loss = self.get_loss(self.model, self.torch_datasets["train"], known_seq_len)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        self.lr = self.scheduler.get_last_lr()[0]
        wandb.log({"train_loss_batch": loss, "known_seq_len": known_seq_len, "lr": self.lr})
        return loss
        

    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.
        with torch.no_grad():
            for known_seq_len in range(1, self.max_len):
                loss = self.get_loss(self.model, self.torch_datasets["val"], known_seq_len)
                total_loss += loss
        return total_loss / (self.max_len - 1)
        # TODO Stop/PAD handling in loss            

    def train(self, epochs: int, val_period: int = 10):
        best_val_loss = float('inf')
        run_path = pathlib.Path("checkpoints", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        run_path.mkdir(parents=True)
        best_model_params_path = os.path.join(run_path, "best_model_params.pt")

        with wandb.init(
            project="WyckoffTransformer",
            job_type="train"):
            wandb.define_metric("epoch")
            wandb.define_metric("val_loss_epoch", step_metric="epoch")
            for epoch in range(1, epochs + 1):
                self.train_step()
                if epoch % val_period == 0:
                    val_loss_epoch = self.evaluate().item()
                    wandb.log({"val_loss_epoch": val_loss_epoch, "epoch": epoch})
                    if val_loss_epoch < best_val_loss:
                        print(f"New best val_loss_epoch {val_loss_epoch}, saving the model")
                        best_val_loss = val_loss_epoch
                        wandb.save(best_model_params_path)
                        torch.save(self.model.state_dict(), best_model_params_path)
                        print(f"Epoch {epoch} val_loss_epoch {val_loss_epoch} saved to {best_model_params_path}")
                self.scheduler.step()
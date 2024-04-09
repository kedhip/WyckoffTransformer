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


class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_len: int = 25):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)[:, :pe.shape[2] // 2]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return x + self.pe[:, :x.size(1)]

class WyckoffTransformerModel(nn.Module):

    def __init__(self, n_space_groups: int, n_sites: int, n_elements: int, max_enumeration: int,
                 d_space_groups: int, d_sites: int, d_species: int, nhead: int, d_hid: int,
                 max_len: int, nlayers: int, dropout: float):
        super().__init__()
        self.model_type = 'Transformer'
        # We are going to concatenate the site, the element, and the space group
        # +1 for inter-ss enumeration
        self.d_model = d_sites + d_species + d_space_groups + 1
        # n_enumeration = max_enumeration + 1
        self.n_token = n_sites + n_elements + max_enumeration + 1
        self.n_elements = n_elements
        self.n_sites = n_sites
        self.max_enumeration = max_enumeration
        self.pos_encoder = PositionalEncoding(self.d_model, max_len)
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout, batch_first=True)
        # Nested tensors are broken in the current version of PyTorch
        # https://github.com/pytorch/pytorch/issues/97111
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, enable_nested_tensor=False)
        self.sites_embedding = nn.Embedding(n_sites, d_sites)
        self.species_embedding = nn.Embedding(n_elements, d_species)
        self.space_groups_embedding = nn.Embedding(n_space_groups, d_space_groups)
        # We predict both the site and the element
        # However, they can not be sampled independently!
        self.linear = nn.Linear(self.d_model, self.n_token)


    def forward(self,
                space_group: Tensor,
                sites: Tensor,
                species: Tensor,
                enumeration: Tensor,
                padding_mask: Tensor) -> (Tensor, Tensor):
        """
        Arguments:
            space_group: Tensor of shape ``[batch_size]``
            sites: Tensor, shape ``[batch_size, seq_len]``
            species: Tensor, shape ``[batch_size, seq_len]``

        Returns:
            Tensor of shape ``[batch_size, seq_len, n_elements]``
            Tensor of shape ``[batch_size, seq_len, n_sites]``
            Tensor of shape ``[batch_size, seq_len, max_enumeration + 1]``
            Remember respect masking when training!
        """
        # And here is a good question on masking
        # 1. Padding. If we just mask it, the model won't be able to stop predicting, we'll need to additinally introduce a stop token.
        # probably, can do without for now. The previous iteration reliably predicted sequences of PADs
        # 2. Casual masking. We can't use it as is, as we need the model to attend on the element of the site
        # 3. While we output the whole sequence, we should only train on the last element, as we use MASK for marking the site under prediction
        # and since we do incremental batching, using the previous elements multiple time would unjustly increase their weight
        # print(sites.shape, species.shape, space_group.shape, padding_mask.shape)
        # print(self.sites_embedding(sites).shape, self.species_embedding(species).shape,
        #      self.space_groups_embedding(torch.tile(space_group.view(-1, 1), (1, sites.shape[1]))).shape)
        src = torch.concat([self.sites_embedding(sites), self.species_embedding(species),
                            enumeration.view(enumeration.shape[0], enumeration.shape[1], 1),
                            self.space_groups_embedding(torch.tile(space_group.view(-1, 1), (1, sites.shape[1])))], dim=2) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)
        output = self.linear(output)
        return (output[:, :, :self.n_elements],
                output[:, :, self.n_elements:self.n_elements + self.n_sites],
                output[:, :, self.n_elements + self.n_sites:])


def get_batches(
    symmetry_sites: Tensor,
    symmetry_elements: Tensor,
    symmetry_enumeration: Tensor,
    element_seq_len: int,
    site_mask: Tensor,
    enumeration_mask: Tensor,
    element_pad: Tensor|None = None,
    site_pad: Tensor|None = None,
    enumeration_pad: Tensor|None = None) -> Tuple[Tensor, Tensor]:
    """
    Args:
        element_seq_len: number of elements considered known
    Returns:
        Tuples of (data, target, padding_mask):
        1. Element probability given i sites and elements
        2. Site symmetry probability given i sites and i+1 elements
        3. Site enumeration probability given i+1 sites and i+1 elements
        
    """
    # Can't predict past the last token
    assert element_seq_len < symmetry_elements.shape[1]
    # Predict element number element_seq_len
    target_element = symmetry_elements[:, element_seq_len]

    data_element = {
        "species": symmetry_elements[:, :element_seq_len],
        "sites": symmetry_sites[:, :element_seq_len],
        "enumeration": symmetry_enumeration[:, :element_seq_len]
    }
    
    # For sites, we condition on species, hence the + 1  in species
    data_site = {
        "species": symmetry_elements[:, :element_seq_len + 1],
        "sites": torch.cat([symmetry_sites[:, :element_seq_len], site_mask.expand(symmetry_sites.shape[0], 1)], dim=1),
        "enumeration": torch.cat([symmetry_enumeration[:, :element_seq_len],
                                  enumeration_mask.expand(symmetry_enumeration.shape[0], 1)], dim=1)
    }
    target_site = symmetry_sites[:, element_seq_len]

    data_enumeration = {
        "species": symmetry_elements[:, :element_seq_len + 1],
        "sites": symmetry_sites[:, :element_seq_len + 1],
        "enumeration": torch.cat([symmetry_enumeration[:, 0:element_seq_len],
                                  enumeration_mask.expand(symmetry_enumeration.shape[0], 1)], dim=1)
    }
    target_enumeration = symmetry_enumeration[:, element_seq_len]

    return ((data_element, target_element), (data_site, target_site), (data_enumeration, target_enumeration))


class WyckoffTrainer():
    def __init__(self,
                 model: nn.Module,
                 torch_datasets: dict,
                 max_len: int,
                 site_mask: Tensor,
                 element_pad: Tensor,
                 site_pad: Tensor):
        # Sequences have difference lengths, so we need to make sure that
        # long sequences don't dominate the loss
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.lr = 5.0
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.9, patience=5)
        self.max_len = max_len
        self.model = model
        self.torch_datasets = torch_datasets
        self.mean_lattice_volume = torch.mean(torch_datasets["train"]["lattice_volume"])
        self.std_latice_volume = torch.std(torch_datasets["train"]["lattice_volume"])
        self.normalised_lattice_volume = dict()
        for dataset_name in torch_datasets:
            self.normalised_lattice_volume[dataset_name] = \
                (torch_datasets[dataset_name]["lattice_volume"] - self.mean_lattice_volume) / self.std_latice_volume
        self.enumeration_mask = torch.tensor([-1], device=site_mask.device)
        self.site_mask = site_mask
        self.element_pad = element_pad
        self.site_pad = site_pad

    def get_token_loss(self, dataset, known_seq_len):
        (data_element, target_element), (data_site, target_site), (data_enumeration, target_enumeration) = get_batches(
            dataset["symmetry_sites"],
            dataset["symmetry_elements"],
            dataset["symmetry_sites_enumeration"],
            known_seq_len,
            self.site_mask,
            self.enumeration_mask)
        output_element = self.model(dataset["spacegroup_number"], **data_element,
                                    padding_mask=dataset["padding_mask"][:, :known_seq_len])[0]
        element_loss_mask = target_element != self.element_pad
        masked_target_element = target_element[element_loss_mask]
        if len(masked_target_element) == 0:
            loss_element = torch.tensor(0., device=masked_target_element.device)
        else:
            loss_element = self.criterion(output_element[element_loss_mask, -1, :], masked_target_element)

        output_site = self.model(dataset["spacegroup_number"], **data_site,
                                 padding_mask=dataset["padding_mask"][:, :known_seq_len + 1])[1]
        # Since we predict STOP element, we need a separate mask for site & enumeration
        site_loss_mask = target_site != self.site_pad
        masked_target_site = target_site[site_loss_mask]
        if len(masked_target_site) == 0:
            loss_site = torch.tensor(0., device=masked_target_element.device)
        else:
            loss_site = self.criterion(output_site[site_loss_mask, -1, :], masked_target_site)
        
        output_enumeration = self.model(dataset["spacegroup_number"], **data_enumeration,
                                        padding_mask=dataset["padding_mask"][:, :known_seq_len + 1])[2]
        # Enumeration mask = enumeration pad = -1
        enumeration_loss_mask = target_enumeration != self.enumeration_mask
        assert (enumeration_loss_mask == site_loss_mask).all()
        masked_target_enumeration = target_enumeration[enumeration_loss_mask]
        if len(masked_target_enumeration) == 0:
            loss_enumeration = torch.tensor(0., device=masked_target_enumeration.device)
        else:
            loss_enumeration = self.criterion(output_enumeration[enumeration_loss_mask, -1, :], masked_target_enumeration)
        return loss_element, loss_site, loss_enumeration

    def train_step(self):
        self.model.train()
        known_seq_len = randint(1, self.max_len - 1)
        loss_element, loss_site, loss_enumeration = self.get_token_loss(self.torch_datasets["train"], known_seq_len)                                  
        loss = loss_element + loss_site + loss_enumeration
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        self.lr = self.scheduler.get_last_lr()[0]
        wandb.log({"train_loss_batch": loss,
                   "train_loss_element": loss_element,
                   "train_loss_site": loss_site,
                   "train_loss_enumeration": loss_enumeration,
                   "known_seq_len": known_seq_len, "lr": self.lr})
        return loss
        
    def evaluate(self) -> Tensor:
        """
        Evaluates the model by calculating the average loss on the validation dataset.

        Returns:
            The average loss on the validation dataset.
        """
        self.model.eval()
        total_loss = 0.
        with torch.no_grad():
            for known_seq_len in range(1, self.max_len - 1):
                loss_token = self.get_token_loss(self.torch_datasets["val"], known_seq_len)
                total_loss += sum(loss_token)
        # ln(P) = ln p(t_n|t_n-1, ..., t_1) + ... + ln p(t_2|t_1)
        # We are minimising the negative log likelihood of the whole sequences
        return total_loss / self.torch_datasets["val"]["symmetry_sites"].shape[0]

    def train(self, epochs: int, val_period: int = 10):
        best_val_loss = float('inf')
        run_path = pathlib.Path("checkpoints", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        run_path.mkdir(parents=True)
        best_model_params_path = run_path / "best_model_params.pt"

        with wandb.init(
            project="WyckoffTransformer",
            job_type="train"):
#            wandb.watch(self.model, criterion=self.criterion)
            for epoch in range(1, epochs + 1):
                self.train_step()
                if epoch % val_period == 0:
                    val_loss_epoch = self.evaluate().item()
                    wandb.log({"val_loss_epoch": val_loss_epoch}, step=epoch)
                    if val_loss_epoch < best_val_loss:
                        best_val_loss = val_loss_epoch
                        wandb.save(str(best_model_params_path))
                        torch.save(self.model.state_dict(), best_model_params_path)
                        print(f"Epoch {epoch} val_loss_epoch {val_loss_epoch} saved to {best_model_params_path}")
                    self.scheduler.step(val_loss_epoch)
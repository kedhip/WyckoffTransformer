from typing import Tuple, List, Optional
import logging
import torch
from torch import nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from omegaconf import OmegaConf


class CascadeEmbedding(nn.Module):
    def __init__(self,
                 cascade,
                 **kwargs):
        """
        Arguments:
            cascade: a tuple of tuples (N_i, d_i, pad_i), where N_i is the number of possible values for the i-th token,
                d_i is the dimensionality of the i-th token embedding. If d_i is None, the token is not embedded.
                pad_i is padding_idx to be passed to torch.nn.Embedding
        """
        super().__init__()
        self.embeddings = torch.nn.ModuleList()
        self.total_embedding_dim = 0
        for n, d, pad in cascade:
            if d is None:
                self.embeddings.append(None)
                self.total_embedding_dim += 1
            else:
                self.embeddings.append(nn.Embedding(n, d, padding_idx=pad, **kwargs))
                self.total_embedding_dim += d


    def forward(self, x: List[Tensor]) -> Tensor:
        """
        Arguments:
            x: Tensor of shape ``[batch_size, seq_len, len(cascade)]``

        Returns:
            Tensor of shape ``[batch_size, seq_len, self.total_embedding_dim]``
        """
        list_of_embeddings = []
        for tensor, emb in zip(x, self.embeddings):
            if emb is None:
                list_of_embeddings.append(tensor.unsqueeze(-1))
            else:
                list_of_embeddings.append(emb(tensor))
        return torch.cat(list_of_embeddings, dim=2)


class CascadeTransformer(nn.Module):
    @classmethod
    def from_config_and_tokenisers(cls, config: OmegaConf,
        tokenisers: dict, device: torch.device):

        if len(config.tokeniser.augmented_token_fields) > 1:
            raise ValueError("Only one augmented field is supported")

        if "cascade_order" in config.model:
            cascade_order = config.model.cascade_order
            logging.info("Using cascade order %s", cascade_order)
        else:
            cascade_order = tuple(config.model.cascade_embedding_size.keys())

        full_cascade = dict()
        for field in cascade_order:
            full_cascade[field] = (len(tokenisers[field]),
                                config.model.cascade_embedding_size[field],
                                tokenisers[field].pad_token)

        return cls(
            n_start=len(tokenisers[config.model.start_token]),
            cascade=full_cascade.values(),
            **config.model.CascadeTransformer_args
            ).to(device)


    def __init__(self,
                 n_start: int,
                 cascade: Tuple[Tuple[int, int|None, int], ...],
                 learned_positional_encoding_max_size: Optional[int],
                 learned_positional_encoding_only_masked: bool,
                 use_token_sum_for_prediction: bool,
                 num_fully_connected_layers: int,
                 TransformerEncoderLayer_args: dict,
                 TransformerEncoder_args: dict):
        """
        Expects tokens in the following format:
        START_k -> [] -> STOP -> PAD
        START_k is the start token, it can take one of the K values, will be embedded.
            In Wychoff transformer, we store the space group here.
        [] is the cascading token. Each element of the token can take values from 0 to N_i - 1.
         During input, some of them are embedded, some not, as per the cascade. We predict the probability for each value, from 
         0 to N_i. Non-embedded values are expected to be floats.
        STOP is stop.
        PAD is padding. Not predicted.

        Arguments:
            n_start: Number of possible start tokens,
            cascade: a tuple of tuples (N_i, d_i, pad_i), where N_i is the number of possible values for the i-th token,
                d_i is the dimensionality of the i-th token embedding. If d_i is None, the token is not embedded.
                pad_i is padding_idx to be passed to torch.nn.Embedding
            n_head: Number of heads in the multiheadattention models. Note that for the concatenated tokens,
                setting it to > 1 may lead to issues, as the hidden dimension is spllit between the heads.
            n_layers: Number of encoder layers in the transformer.
            d_hid: Dimensionality of the hidden layer in the transformer.
            dropout: Dropout value.
        """
        super().__init__()
        self.embedding = CascadeEmbedding(cascade)
        self.d_model = self.embedding.total_embedding_dim
        self.encoder_layers = TransformerEncoderLayer(self.d_model, batch_first=True, **TransformerEncoderLayer_args)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, **TransformerEncoder_args)
        self.start_embedding = nn.Embedding(n_start, self.d_model)
        self.learned_positional_encoding_max_size = learned_positional_encoding_max_size
        self.learned_positional_encoding_only_masked = learned_positional_encoding_only_masked
        if learned_positional_encoding_max_size != 0:
            self.positions_embedding = nn.Embedding(
                learned_positional_encoding_max_size,
                self.d_model)
        # Since our tokens are concatenated, we need to mix the embeddings
        # before we can use multuple attention heads.
        # Actually, a fully-connected layer is an overparametrisation
        # but it's easier to implement. Completely redundant if nhead == 1.
        self.mixer = nn.Linear(self.d_model, self.d_model, bias=False)
        # Note that in the normal usage, we want to condition the cascade element prediction
        # on the previous element, so care should be taken as to which head to call.
        self.use_token_sum_for_prediction = use_token_sum_for_prediction
        prediction_head_size = 2 * self.d_model if use_token_sum_for_prediction else self.d_model
        self.prediction_heads = torch.nn.ModuleList()
        if num_fully_connected_layers == 0:
            raise ValueError("num_fully_connected_layers must be at least 1 for dimensionality reasons.")
        for output_size, _, _ in cascade:
            if num_fully_connected_layers == 1:
                self.prediction_heads.append(nn.Linear(prediction_head_size, output_size))
            else:
                this_sequence = []
                for _ in range(num_fully_connected_layers - 1):
                    this_sequence.append(nn.Linear(prediction_head_size, prediction_head_size))
                    this_sequence.append(nn.ReLU())
                this_sequence.append(nn.Linear(prediction_head_size, output_size))
                self.prediction_heads.append(nn.Sequential(*this_sequence))


    def forward(self,
                start: Tensor,
                cascade: List[Tensor],
                padding_mask: Tensor,
                prediction_head: int) -> Tensor:
        logging.debug("Cascade len: %i", len(cascade))
        cascade_embedding = self.embedding(cascade)
        logging.debug("Cascade reported embedding dim: %i", self.embedding.total_embedding_dim)
        logging.debug("Cascade embedding size: (%i, %i, %i)", *cascade_embedding.size())
        cascade_embedding = self.mixer(cascade_embedding)
        if self.learned_positional_encoding_max_size != 0:
            if self.learned_positional_encoding_only_masked:
                positional_encoding = self.positions_embedding(
                    torch.tensor([cascade_embedding.size(1) - 1], device=start.device, dtype=start.dtype))
                cascade_embedding[:, -1] += positional_encoding
            else:
                sequence_range = torch.arange(0, cascade_embedding.size(1), device=start.device, dtype=start.dtype)
                positional_encoding = self.positions_embedding(sequence_range)
                cascade_embedding += positional_encoding.unsqueeze(0)

        data = torch.cat([self.start_embedding(start).unsqueeze(1), cascade_embedding], dim=1)
        transformer_output = self.transformer_encoder(data, src_key_padding_mask=padding_mask)
        if self.use_token_sum_for_prediction:
            prediction_input = torch.cat([transformer_output[:, -1], transformer_output[:, :-1].sum(dim=1)], dim=1)
        else:
            prediction_input = transformer_output[:, -1]
        prediction = self.prediction_heads[prediction_head](prediction_input)
        return prediction

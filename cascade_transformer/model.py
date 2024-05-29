from typing import Tuple, List
import torch
from torch import nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
    def __init__(self,
                 n_start: int,
                 cascade: Tuple[Tuple[int, int|None, int], ...],
                 n_head: int,
                 n_layers: int,
                 d_hid: int,
                 dropout: float,
                 use_mixer: bool=True):
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
        self.encoder_layers = TransformerEncoderLayer(self.d_model, n_head, d_hid, dropout, batch_first=True)
        # Nested tensors are broken in the current version of PyTorch
        # https://github.com/pytorch/pytorch/issues/97111
        # We also don't need them as we ensure that batches all have the same langth in WychoffTrainer
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, n_layers, enable_nested_tensor=False)
        self.start_embedding = nn.Embedding(n_start, self.d_model)
        # So that multiple heads can be used
        self.use_mixer = use_mixer
        if not use_mixer:
            raise NotImplementedError("Only mixer=True is supported")
        # Since our tokens are concatenated, we need to mix the embeddings
        # before we can use multuple attention heads.
        # Actually, a fully-connected layer is an overparametrisation
        # but it's easier to implement
        self.mixer = nn.Linear(self.d_model, self.d_model, bias=False)
        # Note that in the normal usage, we want to condition the cascade element prediction
        # on the previous element, so care should be taken as to which head to call.
        self.prediction_heads = torch.nn.ModuleList([nn.Linear(self.d_model, n) for n, _, _ in cascade])
        self.n_head = n_head
        self.n_layers = n_layers
        self.d_hid = d_hid
        self.dropout = dropout


    def forward(self,
                start: Tensor,
                cascade: List[Tensor],
                padding_mask: Tensor,
                prediction_head: int) -> Tensor:
        data = torch.cat([self.start_embedding(start).unsqueeze(1), self.embedding(cascade)], dim=1)
        data = self.mixer(data)
        transformer_output = self.transformer_encoder(data, src_key_padding_mask=padding_mask)
        prediction = self.prediction_heads[prediction_head](transformer_output[:, -1])
        return prediction

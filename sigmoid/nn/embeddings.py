#-----------------------------------------------------------------------------
# converted to PyTorch by DeepseekR1, and adapted accordingly.
# original implementation
#
#  repo: https://github.com/aruberts/TabTransformerTF
#  file: tabtransformertf/models/embeddings.py
#
# Modified by P. Fluxá (Thoughtworks Chile SPA, 2025)
#
#-----------------------------------------------------------------------------
from typing import List, Tuple, Union, Any, Dict

import math

import numpy as np

from einops import rearrange

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class PLE(nn.Module):

    @staticmethod
    def mkbins(
        data: np.ndarray,
        as_dict: bool = False) -> Dict[str, Any]:

        data = np.atleast_2d(data).T

        bins = []
        n_bins = []

        for x in data:
            # use Freedman-Diaconis rule
            iqr = np.subtract(*np.percentile(x, [75, 25]))
            print(iqr)
            bw = 2.0 * iqr / np.cbrt(len(x))
            nb = int(np.ceil((x.max() - x.min()) / bw))
            dx = 1.0 / nb
            intr = np.arange(0, 1 + dx, dx)
            b = torch.unique(
                torch.tensor([
                    np.quantile(x, min([q, 1.0]), method='inverted_cdf') for q in intr
                    ],
                    dtype=torch.float32
                )
            )
            n_bins.append(len(b) - 1)
            bins.append(b)

        d_out = {
            'n_bins': n_bins,
            'bins': bins
        }
        return d_out

    def __init__(self, n_bins: int, bins: torch.Tensor):

        super(PLE, self).__init__()
        self.n_bins = n_bins
        self.register_buffer('bins', bins)

    def forward(self, x):
        """
        Forward pass of the PLE embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (n_features, batch_size).

        Returns:
            torch.Tensor: Output tensor of shape (n_bins, batch_size)
        """
        ple_encoding_one = torch.ones((x.shape[-1], self.n_bins), device=x.device)
        ple_encoding_zero = torch.zeros((x.shape[-1], self.n_bins), device=x.device)

        left_masks = []
        right_masks = []
        other_case = []

        for i in range(1, self.n_bins + 1):
            left_mask = (x < self.bins[i - 1]) & (i > 1)
            right_mask = (i < self.n_bins) & (x >= self.bins[i])
            v = (x - self.bins[i - 1]) / (self.bins[i] - self.bins[i - 1])
            left_masks.append(left_mask)
            right_masks.append(right_mask)
            other_case.append(v)

        left_masks = torch.swapaxes(torch.stack(left_masks, dim=0), 0, 1)
        right_masks = torch.swapaxes(torch.stack(right_masks, dim=0), 0, 1)
        other_case = torch.swapaxes(torch.stack(other_case, dim=0), 0, 1)

        other_mask = right_masks == left_masks  # both are false
        other_case = other_case.float()

        enc = torch.where(left_masks, ple_encoding_zero, ple_encoding_one)
        enc = torch.where(other_mask, other_case, enc)
        enc = torch.reshape(enc, (self.n_bins, -1))

        return enc

class NumericalEmbeddingPLE(nn.Module):

    padval_: int = -1

    def __init__(
        self,
        n_features: int,
        n_bins: List[int], bins: List[torch.Tensor],
        min_emb_dim: int = 2048,
        max_emb_dim: int = -1):

        super(NumericalEmbeddingPLE, self).__init__()

        self.num_features = n_features
        # compute min/max embedding dimensions
        self.min_emb_dim_ = min([min_emb_dim, min(n_bins)])
        self.max_emb_dim_ = max([max_emb_dim, max(n_bins)])
        assert self.min_emb_dim_ > 0, "Minimum embedding dimension must be positive"

        # Initialise embedding layers
        self.embedding_layers_ = nn.ModuleList()
        for idx in range(self.num_features):
            ple = PLE(n_bins[idx], bins[idx])
            self.embedding_layers_.append(ple)

    def forward(self, x):

        embeddings = []
        for i, layer in enumerate(self.embedding_layers_):
            embeddings.append(layer(x[i]))
        padded_emb_tensor = pad_sequence(embeddings, padding_value=self.padval_)
        padded_emb_tensor = rearrange(padded_emb_tensor, 'n s b -> s b n')
        padding_mask = padded_emb_tensor == self.padval_
        padding_mask = rearrange(padding_mask, 's b n-> b s n')[:, :, 0]

        return padded_emb_tensor, padding_mask

class CategoricalEmbedding(nn.Module):

    # default padding value
    padval_: int = -1

    def __init__(
        self,
        cardinalities: List[int],
        min_emb_dim: int = 2048,
        max_emb_dim: int = -1,
    ):
        super(CategoricalEmbedding, self).__init__()
        # compute min/max embedding dimensions
        self.min_emb_dim_ = min([min_emb_dim, min(cardinalities) + 1])
        self.max_emb_dim_ = max([max_emb_dim, max(cardinalities) + 1])

        assert self.min_emb_dim_ > 0, "Minimum embedding dimension must be positive"

        self.embedding_layers_ = nn.ModuleList()
        for c in cardinalities:
            self.embedding_layers_.append(nn.Embedding(c, c + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: categorical data with shape (n_categories, batch_size)
        """
        embeddings = []
        for i, layer in enumerate(self.embedding_layers_):
            emb = rearrange(layer(x[:, i]), 'b c -> c b')
            # emb = layer(x[i])
            embeddings.append(emb)
        padded_emb_tensor = pad_sequence(embeddings, padding_value=self.padval_)
        padding_mask = padded_emb_tensor == self.padval_
        padded_emb_tensor = rearrange(padded_emb_tensor, 'd s b -> s b d')
        padding_mask = rearrange(padding_mask, 'd s b-> b s d')[:, :, 0]

        return padded_emb_tensor, padding_mask

class PositionalEncoding(torch.nn.Module):

    def __init__(self, emb_dim: int, max_len: int = 5000):

        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim)
        )

        pe = torch.zeros(1, max_len, emb_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

def test_numericalPLE_embedding(use_cuda: bool = False):

    batch_size = 256
    n_features = 2
    rng = np.random.default_rng()
    x1 = rng.normal(0.0, 0.1, size=(1, batch_size))
    x2 = rng.normal(0.0, 10.0, size=(1, batch_size))
    x = np.concatenate([x1, x2], axis=0)
    n_bins, bins = PLE.mkbins(x).values()

    num_emb = NumericalEmbeddingPLE(n_features, n_bins, bins)
    if use_cuda:
        num_emb.cuda()

    t = np.asarray([
        [-1.0, -1.0],
        [-0.5, -0.5],
        [ 0.0,  0.0],
        [+0.5, -0.5],
        [+1.0, -1.0]])
    y = torch.tensor(t).t()
    print("numerical input shape:", y.shape)
    if use_cuda:
        y = y.cuda()
    z, m = num_emb(y)

    return z, m

def test_categorical_embedding(use_cuda: bool = False):

    cat_emb = CategoricalEmbedding([4, 6, 8])
    if use_cuda:
        cat_emb.cuda()
    t = np.asarray([
        [0, 3, 5],
        [1, 0, 7],
        [2, 5, 2],
        [3, 4, 6]]
    )
    y = torch.tensor(t).t()
    if use_cuda:
        y = y.cuda()
    print("category input shape:", y.shape)
    z, m = cat_emb(y.long())

    return z, m

if __name__ == "__main__":

    with torch.no_grad():
        zc, mc = test_categorical_embedding()
        print("-" * 79)
        print(zc.shape)
        print(zc)
        print(mc)
        print("-" * 79 + '\n')

        zn, mn = test_numericalPLE_embedding()
        print("-" * 79)
        print(zn.shape)
        print(zn)
        print(mn)
        print("-" * 79 + '\n')
        # print(zc.shape)
        # z = torch.cat([zn, zc], dim=1)
        # pos_enc = PositionalEncoding(16)
        # zp = pos_enc(z)

        # print(zp.shape)

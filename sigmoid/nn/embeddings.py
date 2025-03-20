#-----------------------------------------------------------------------------
# converted to PyTorch by DeepseekR1, and adapted accordingly.
# original implementation 
# 
#  repo: https://github.com/aruberts/TabTransformerTF
#  file: tabtransformertf/models/embeddings.py
#
#-----------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PLE(nn.Module):
    
    def __init__(self, n_bins=10):
        
        super(PLE, self).__init__()
        self.n_bins = n_bins

    def adapt(self, data):
        
        interval = 1 / self.n_bins
        bins = torch.unique(
            torch.tensor(
                [np.quantile(data, q, method='inverted_cdf') for q in np.arange(0, 1 + interval, interval)],
                dtype=torch.float32
            )
        )
        self.n_bins = len(bins) - 1
        self.register_buffer('bins', bins)

    def forward(self, x):
       
        ple_encoding_one = torch.ones((x.shape[1], self.n_bins), device=x.device)
        ple_encoding_zero = torch.zeros((x.shape[1], self.n_bins), device=x.device)

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

        left_masks = torch.transpose(torch.stack(left_masks, dim=1), 1, 2)
        right_masks = torch.transpose(torch.stack(right_masks, dim=1), 1, 2)
        other_case = torch.transpose(torch.stack(other_case, dim=1), 1, 2)

        other_mask = right_masks == left_masks  # both are false
        other_case = other_case.float()
        
        enc = torch.where(left_masks, ple_encoding_zero, ple_encoding_one)
        enc = torch.reshape(torch.where(other_mask, other_case, enc), (-1, 1, self.n_bins))

        return enc

class NumEmbeddingPLE(nn.Module):
    
    def __init__(
        self,
        feature_names: list,
        X: np.array,
        emb_dim: int = 32,
        n_bins: int = 10):
        
        super(NumEmbeddingPLE, self).__init__()

        self.num_features = len(feature_names)
        self.features = feature_names
        self.emb_dim = emb_dim
        
        # Initialise embedding layers
        self.embedding_layers = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        for i, f in enumerate(feature_names):
            emb_l = PLE(n_bins)
            emb_l.adapt(X[:, i])
            
            lin_l = nn.Linear(emb_l.n_bins, emb_dim)
            nn.init.kaiming_normal_(lin_l.weight, mode='fan_in', nonlinearity='relu')
            
            self.embedding_layers[f] = emb_l
            self.linear_layers[f] = lin_l
    
    def embed_column(self, f, data):
       
        emb = self.linear_layers[f](self.embedding_layers[f](data))
        
        return emb
   
    def forward(self, x):
        
        emb_columns = []
        for i, f in enumerate(self.features):
            emb_columns.append(self.embed_column(f, x[:, i]))
        embs = torch.cat(emb_columns, dim=1)
            
        return embs

class CategoryEmbedding(nn.Module):
    def __init__(
        self,
        feature_names: list,
        X: np.array,
        emb_dim: int = 32,
    ):
        super(CategoryEmbedding, self).__init__()
        self.features = feature_names
        self.emb_dim = emb_dim
        
        self.category_prep_layers = nn.ModuleDict()
        self.emb_layers = nn.ModuleDict()
        for i, c in enumerate(self.features):
            unique_values = np.unique(X[:, i])
            lookup = {v: idx for idx, v in enumerate(unique_values)}
            emb = nn.Embedding(len(unique_values), self.emb_dim)

            self.category_prep_layers[c] = lookup
            self.emb_layers[c] = emb
    
    def embed_column(self, f, data):
        lookup = self.category_prep_layers[f]
        indices = torch.tensor([lookup[val] for val in data], dtype=torch.long)
        return self.emb_layers[f](indices)

    def forward(self, x):
        emb_columns = []
        for i, f in enumerate(self.features):
            emb_columns.append(self.embed_column(f, x[:, i]))
        
        embs = torch.stack(emb_columns, dim=1)
        return embs

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

if __name__ == '__main__':
    
    x = np.random.randn(4, 3, 1000)
    x = torch.tensor(x)
    num_emb = NumEmbeddingPLE(['a', 'b', 'c'], x)
    
    t = np.asarray([[[-1.0], [0.0], [1.0]]])
    y = torch.tensor(t)
    z = num_emb(y)
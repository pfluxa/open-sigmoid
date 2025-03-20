
from os import read
import numpy as np
import torch
from einops import rearrange
from torch import nn

from typing import Dict, List, NamedTuple, Optional, Tuple
from typing import Callable, Union, Any, TypeVar

from sigmoid.nn.efficient_net import EfficientNet1D, InverseEfficientNet1D


class Autoembedder(nn.Module):

    def get_autoencoder(self) -> Tuple[nn.Sequential, nn.Sequential, int]:
        """
        Args:
            num_cont_features (int): Number of continues features.
        Returns:
            Tuple[torch.nn.Sequential, torch.nn.Sequential]: Tuple containing the encoder and decoder.
        """

        n_features = 0
        for t in self.embedding_sizes_:
            n_features += t[1]
        print(f"n features: {n_features}")
        codec_dim = self.config["codec_dim"]
        # width_mult = self.config.get('width_multiplier', 1.0)
        # encoder = EfficientNet1D(n_channels=n_features, output_dim=codec_dim, width_mult=width_mult)
        # decoder = InverseEfficientNet1D(
        #     in_features=codec_dim, out_channels=n_features,
        #     input_dim=codec_dim, output_dim=self.embsz_,
        #     width_mult=width_mult
        # )
        encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 1000),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1000, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, codec_dim),
        )
        decoder = torch.nn.Sequential(
            torch.nn.Linear(codec_dim, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 1000),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1000, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, n_features),
        )
        return encoder, decoder, n_features

    def __init__(
        self,
        config: Dict,
        num_cont_features: int,
        embedding_sizes: List[Tuple[int, int]],
    ) -> None:
        """
        Args:
            config (Dict[str, Any]): Configuration for the model.
                In the [documentation](https://chrislemke.github.io/autoembedder/#parameters) all possible parameters are listed.
            num_cont_features (int): Number of continues features.
            embedding_sizes (Optional[List[Tuple[int, int]]]): List of tuples.
                Each tuple contains the size of the dictionary (unique values) of embeddings and the size of each embedding vector.
                Only needs to be provided if categorical columns are used.

        Returns:
            None
        """
        super().__init__()

        self.embedding_sizes_ = embedding_sizes
        self.config = config
        self.last_target: Optional[torch.Tensor] = None
        self.code_value: Optional[torch.Tensor] = None
        self.embeddings = nn.ModuleList([
                nn.Sequential(
                    nn.Embedding(t[0], t[1]),
                    nn.Dropout(p=0.1),
                    nn.BatchNorm1d(t[1]),
                )
                for t in embedding_sizes[1:]
            ]
        )
        self.encoder, self.decoder, n_features = self.get_autoencoder()
        self.n_features_ = n_features
        self.num_embeddings_ = torch.nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_sizes[0][0], embedding_sizes[0][1], bias=False),
                nn.BatchNorm1d(embedding_sizes[0][1]),
            )
        ])
        self.logvar_ = torch.nn.Linear(
            config['codec_dim'], config['codec_dim']
        )
        self.mu_ = torch.nn.Linear(
            config['codec_dim'], config['codec_dim']
        )
        self.sigmoid1_ = nn.Sigmoid()
        self.sigmoid2_ = nn.Sigmoid()
        self.temperature_ = nn.Parameter(torch.tensor(1.0))

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.get("activation", "tanh") == "tanh":
            return nn.Tanh()(x)
        if self.config.get("activation", "tanh") == "relu":
            return nn.ReLU()(x)
        if self.config.get("activation", "tanh") == "leaky_relu":
            return nn.LeakyReLU()(x)
        if self.config.get("activation", "tanh") == "elu":
            return nn.ELU()(x)
        raise ValueError(
            f"""
            Unsupported activation: `{self.config['activation']}`!.
            Please pick one of the following: `tanh`, `relu`, `leaky_relu`, `elu`.
            """
        )

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cat (torch.Tensor): Tensor including the categorical values. Shape: [columns count, batch size]
            x_cont (torch.Tensor): Tensor including the continues values. Shape: [columns count, batch size]
        Returns:
            torch.Tensor :Output of the 'Autoembedder'. It contains the concatenated and processed continues and categorical data.
        """
        # x_cont = rearrange(x_cont, 'c b -> b c')
        x_emb = []
        for i, layer in enumerate(self.num_embeddings_):
            y = layer(x_cont)
            x_emb.append(y)

        for i, layer in enumerate(self.embeddings):
            value = x_cat[i].int()
            y = layer(value)
            x_emb.append(y)

        x = torch.cat(x_emb, 1)
        last_target = (
            x.clone().detach()
        )  # Concatenated x values - used with the custom loss function: `AutoEmbLoss`.
        x = self.encoder(x)
        l = self.logvar_(x)
        u = self.mu_(x)
        s = self.sigmoid1_(x)

        z1 = self.reparameterize(u, l)

        eps = torch.rand_like(s)
        z2 = self.sigmoid2_(torch.log(s + 1e-7) - torch.log(-torch.log(eps + 1e-7) + 1e-7))

        x = z2 * x + z1
        code_value = x.clone().detach()  # Stores the values of the code layer.

        x = self.decoder(x)

        return x, last_target, code_value , u, l, z2

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def init_xavier_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

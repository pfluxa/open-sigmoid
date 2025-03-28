
from os import read, system
from types import CodeType

from networkx.algorithms.hybrid import kl_connected_subgraph
import numpy as np
import torch
from einops import rearrange
from torch import nn

from typing import Dict, List, NamedTuple, Optional, Tuple
from typing import Callable, Union, Any, TypeVar

from sigmoid.nn.embeddings import ZIELEmbeddings
from sigmoid.nn.embeddings import CategoricalEmbedding


class Autoembedder(nn.Module):

    def __init__(self, architecture: Dict[str, Any]) -> None:
        """
        Args:
            architecture (Dict[str, Any]): Configuration for the model.
                In the [documentation](https://chrislemke.github.io/autoembedder/#parameters) all possible parameters are listed.
            embedding_sizes (List[Tuple[int, int]]): List of tuples.
                Each tuple contains the size of the dictionary (unique values) of embeddings and the size of each embedding vector.
                Only needs to be provided if categorical columns are used.

        Returns:
            None
        """
        super(Autoembedder, self).__init__()

        self.config = {}
        for k, v in architecture.items():
            self.config[k] = v

        self.hidden_dim_ = -1
        self.codec_dim_ = architecture['codec_dim']
        self.n_features_ = -1
        self.num_last_target_: Optional[torch.Tensor] = None
        self.cat_last_target_: Optional[torch.Tensor] = None
        self.code_value_: Optional[torch.Tensor] = None

        self.num_emb_ = nn.Module
        self.enc_num_ = nn.Module

        self.cat_emb_ = nn.Module
        self.enc_cat_ = nn.Module

        self.decoder_ = nn.Module

        self.d_num_ = -1
        self.c_num_ = -1

    def build_embedding_layers(
        self,
        feature_ranges: List[Tuple[float, float]],
        max_depths: List[int],
        cardinalities: List[int],
    ):
        """ Build the embedding layers.
        """
        self.n_num_ = len(feature_ranges)
        self.n_cat_ = len(cardinalities)
        # numerical embedding
        num_embeddings = ZIELEmbeddings(feature_ranges, max_depths)
        # categorical embeddings
        cat_embeddings = CategoricalEmbedding(cardinalities)

        self.num_emb_ = num_embeddings
        self.d_num_ = num_embeddings.max_emb_dim_
        src_mask = torch.zeros((self.n_num_, self.n_num_), dtype=torch.bool)
        self.register_buffer('num_src_mask_', src_mask)

        self.cat_emb_ = cat_embeddings
        self.d_cat_ = cat_embeddings.max_emb_dim_
        src_mask = torch.zeros((self.n_cat_, self.n_cat_), dtype=torch.bool)
        self.register_buffer('cat_src_mask_', src_mask)

        self.num_pooling_ = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                self.d_num_ * self.n_num_,
                self.codec_dim_
            ),
        )
        self.cat_pooling_ = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                self.d_cat_ * self.n_cat_,
                self.codec_dim_
            ),
        )
        self.codec_pooling_ = nn.Linear(2 * self.codec_dim_, self.codec_dim_)

    def build_decoder(self):
        """
        Build the decoder model.
        """
        decoder_block = nn.ModuleList()
        # create decoder
        hidden_dims = self.config['decoder_dims']
        for i in range(0, len(hidden_dims)):
            layer = nn.Linear(
                hidden_dims[i]['in_dim'],
                hidden_dims[i]['out_dim'],
                bias=False
            )
            decoder_block.append(layer)
        decoder = nn.Sequential(*decoder_block)

        self.num_unpooling_ = nn.Sequential(
            nn.Linear(hidden_dims[-1]['out_dim'], self.d_num_),
            nn.Sigmoid()
        )
        self.cat_unpooling_ = nn.Linear(hidden_dims[-1]['out_dim'], self.d_cat_)
        self.decoder_ = decoder

    def build_cat_encoder(self) -> None:
        """ Build the categorical encoder model.
        """
        print(f"building encoder layer with d_model = {self.d_cat_}")
        encoder_block = nn.TransformerEncoderLayer(
            d_model=self.d_cat_,
            nhead=1,
            dim_feedforward=512,
        )
        encoder = nn.TransformerEncoder(
            encoder_block, num_layers=4,
            enable_nested_tensor=False
        )

        self.cat_encoder_ = encoder

    def build_num_encoder(self) -> None:
        """ Build the numerical encoder model.
        """
        print(f"building encoder layer with d_model = {self.d_num_}")
        encoder_block = nn.TransformerEncoderLayer(
            d_model=self.d_num_,
            nhead=1,
            dim_feedforward=512,
        )
        encoder = nn.TransformerEncoder(
            encoder_block, num_layers=4,
            enable_nested_tensor=False
        )

        self.num_encoder_ = encoder

    def forward(self,
            x_cat: torch.Tensor,
            x_cont: torch.Tensor,
            mask_idx_num: Optional[int] = None,
            mask_idx_cat: Optional[int] = None
        ) -> Tuple[torch.Tensor,
                   torch.Tensor,
                   torch.Tensor,
                   torch.Tensor,
                   torch.Tensor,
                   torch.Tensor]:
        """
        Args:
            x_cat (torch.Tensor): Tensor including the categorical values. Shape: [columns count, batch size]
            x_cont (torch.Tensor): Tensor including the continues values. Shape: [columns count, batch size]
        Returns:
           torch.Tensor :Output of the 'Autoembedder'. It contains the concatenated and processed continues and categorical data.
        """
        cat_emb, cat_key_msk = self.cat_emb_(x_cat)
        cat_msk = self.cat_src_mask_.clone().detach()
        cat_msk[mask_idx_cat] = True
        cat_last_target = (cat_emb.clone().detach())[mask_idx_cat]

        num_emb, num_key_msk = self.num_emb_(x_cont)
        num_msk = self.num_src_mask_.clone().detach()
        num_msk[mask_idx_num] = True
        num_last_target = (num_emb.clone().detach())[mask_idx_num]

        # print("cat emb shape:", cat_emb.shape)
        # print("cat msk shape:", cat_msk.shape)
        # print("cat msk key shape:", cat_key_msk.shape)
        cat_h = self.cat_encoder_(cat_emb, mask=cat_msk, src_key_padding_mask=cat_key_msk)
        #print(cat_h[:, 0, :])
        cat_h = rearrange(cat_h, 's b n -> b s n')
        cat_z = self.cat_pooling_(cat_h)

        # print("num emb shape:", num_emb.shape)
        # print("num msk shape:", num_msk.shape)
        # print("num msk key shape:", num_key_msk.shape)
        num_h = self.num_encoder_(num_emb, mask=num_msk, src_key_padding_mask=num_key_msk)
        # print(num_h[:, 0, :])
        num_h = rearrange(num_h, 's b n -> b s n')
        num_z = self.num_pooling_(num_h)

        z_nc = torch.cat([num_z, cat_z], dim=1)
        z = self.codec_pooling_(z_nc)

        sec_loss = self.spherical_embedding_loss(z)
        code_value = z.clone().detach()

        w = self.decoder_(z)
        num_x = self.num_unpooling_(w)
        cat_x = self.cat_unpooling_(w)

        return (
            num_x, cat_x,
            num_last_target, cat_last_target,
            code_value,
            sec_loss)

    def spherical_embedding_loss(self, x):

        xn = x.norm(p=2, dim=1)
        xn_mean = xn.mean().detach()
        self.norm_mean_ = xn_mean
        xn_var = xn.var()
        l_sec = xn - xn_mean
        l_sec = (torch.sqrt(xn_var + l_sec * l_sec)).mean()# / x.shape[0]

        return l_sec

    def kld_bernoulli(self, p):

        kld_loss = torch.mul(p,
            torch.log(p + 1e-10) - torch.log(self.prior_)
        ) + torch.mul(1 - p, torch.log(1 - p + 1e-10) - torch.log(1 - self.prior_))
        kld_loss = kld_loss.sum() / p.shape[0]

        return kld_loss

    def kld_normal(self, mu, logvar):

        kld_loss = 1 + logvar - torch.pow(mu, 2) - torch.exp(logvar)
        kld_loss = kld_loss.sum() / mu.shape[0]
        kld_loss *= -0.5

        return kld_loss

    def reparam_bernoulli(self, p: torch.Tensor, temperature: Optional[torch.Tensor] = torch.tensor(0.1)) -> torch.Tensor:
        """
        Reparameterization trick to sample from Bernoulli(logits).

        :param p: (Tensor)
            Strictly positive output from previous layer
        :param temperature: (Optional, Tensor)
            Set to ~5 to obtain random outputs. Defaults to 0.1.
        :return: (Tensor) [B x D]

        :note
            Reparameterization:
            -- Let u be a uniform random variale in [0,1], p be the predicted probability (i.e. input),
            -- let l be the temperature.
            -- y = sigmoid((log(p) + log(u) - log(1 - u))/l)

        :note
            The reparameterization trick assumes that the next layer is a Sigmoid layer
        """
        u = torch.rand_like(p)
        y = (torch.log(p) + torch.log(u) - torch.log(1 - u)) / temperature
        return y

    def reparam_normal(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
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

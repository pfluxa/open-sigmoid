import time

from typing import Dict, List, NamedTuple, Optional, Tuple

from tqdm import tqdm

from einops import rearrange

import torch
from torch.utils.data import DataLoader

from torcheval.metrics import MeanSquaredError

from sigmoid.nn.utils import GracefulExiter
from sigmoid.nn.embedders import Autoembedder


class AutoEmbedderWrapper(torch.nn.Module):
    """ Auto-encoder that trains itself on mixed types of data.
    """
    def __init__(self, config: Dict, n_numerical: int, embedding_sizes: List[Tuple[int, int]]):

        super(AutoEmbedderWrapper, self).__init__()

        self.config_ = config
        self.n_numeric_ = n_numerical
        self.n_catg_ = len(embedding_sizes)
        self.embsz_ = embedding_sizes
        model = Autoembedder(self.config_, self.n_numeric_, self.embsz_)
        self.in_features_ = model.n_features_
        self.model_ = model # torch.nn.DataParallel(model)
        self.reconstruction_loss_ = torch.nn.MSELoss(reduction='mean')
        self.prior = torch.tensor(0.5)

        self.metric_ = {}
        self.metric_['mse'] = []
        self.mse_ = MeanSquaredError()
        self.seed_ = 42
        self.compute_device_ = torch.device

    def set_device(self, device: torch.device):
        """ Sets the device to run training/inference.

            @param device: torch.device
                instance of torch.device (ex: torch.device('cpu'))
        """
        self.compute_device_ = device
        self.model_.to(self.compute_device_)
        self.mse_ = self.mse_.to(self.compute_device_)
        self.prior = self.prior.to(self.compute_device_)

    def set_kld_weight(self, kld_weight: float):
        """ Sets the weight of the KLD loss.

            @param kld_weight: float
                weight of the KLD loss
        """
        self.kld_weight_ = kld_weight

    def get_device(self) -> torch.device:
        """ Returns device used for inference and training.
        """
        return self.compute_device_

    def get_n_input(self):

        return self.in_features_

    def get_encodings(self) -> torch.Tensor:
        """ Returns encodings of last mini-batch pass.
        """
        return self.code_value_

    def get_embedded_input(self,  transposed: bool = False) -> torch.Tensor:
        """ Returns input embeddings of last mini-batch pass.
        """
        e = self.last_target_
        if transposed:
            e = torch.swapaxes(e, 1, 2)
        return e

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """ Overloads forward method.
        """
        x, lt, ct, u, l = self.model_(x_cat, x_cont)
        # h_loss = torch.mul(b, torch.log(b + 1e-20) - torch.log(self.prior)) \
        #        + torch.mul(1 - b, torch.log(1 - b + 1e-20) - torch.log(1 - self.prior))
        kld_loss = torch.mean(
            (-0.5 * torch.sum(1 + l - u ** 2 - l.exp(), dim = 2)).sum(dim = 1), dim=0
        )
        self.last_target_ = lt
        self.code_value_ = ct

        # return x, kld_loss, torch.mean(h_loss.sum(dim=1), dim=0)
        return x, kld_loss

    def train_single_epoch(self,
                           dataloader: DataLoader,
                           optimizer: torch.optim.Optimizer,
                           n_params):
        """ Train auto-encoder for a single epoch.

            @param
        """
        self.train(True)
        epoch_loss = 0.0
        for x_num, x_cat in tqdm(dataloader):
            # x_num = rearrange(x_num, 'r c -> c r')
            x_cat = rearrange(x_cat, 'r c -> c r')
            x_num = x_num.to(self.compute_device_)
            x_cat = x_cat.to(self.compute_device_)
            # x_num = torch.swapaxes(0, 1)
            # every batch starts with zero gradients
            optimizer.zero_grad()
            # run autoencoder on batch
            with torch.autocast(device_type="cuda"):
                # x_emb_hat, kld_loss, h_loss = self(x_num, x_cat)
                x_emb_hat, kld_loss = self(x_num, x_cat)
                x_emb = self.get_embedded_input()
                reconstruction_loss = self.reconstruction_loss_(x_emb_hat, x_emb)
                regularization_loss = sum(p.abs().sum() for p in self.parameters()) / n_params
                loss = reconstruction_loss + (kld_loss + 0.0001 * regularization_loss) * self.kld_weight_
            # print(regularization_loss.item())
            # adjust learning weights
            loss.backward()
            # update uptimizer
            optimizer.step()
            # update running (training) loss
            epoch_loss += loss.item()
        self.train(False)

        return epoch_loss

    def eval_single_epoch(self, dataloader: DataLoader, n_params: int):
        """ Runs model in validation data.
        """
        self.eval()
        testing_loss = 0.0
        self.mse_.reset()

        with torch.no_grad():
            for x_num, x_cat in tqdm(dataloader):
                # x_num = rearrange(x_num, 'r c -> c r')
                x_cat = rearrange(x_cat, 'r c -> c r')
                x_num = x_num.to(self.compute_device_)
                x_cat = x_cat.to(self.compute_device_)
                # run autoencoder on batch
                # x_emb_hat, kld_loss, h_loss = self(x_num, x_cat)
                x_emb_hat, kld_loss = self(x_num, x_cat)
                x_emb = self.get_embedded_input()
                reconstruction_loss = self.reconstruction_loss_(x_emb_hat, x_emb)
                regularization_loss = sum(p.abs().sum() for p in self.parameters()) / n_params
                # loss = reconstruction_loss + (h_loss + kld_loss + 0.0001 *regularization_loss) * self.kld_weight_
                loss = reconstruction_loss + (kld_loss + 0.0001 *regularization_loss) * self.kld_weight_
                # update running (training) loss
                testing_loss += loss.item()
                # compute metrics
                self.mse_.update(
                    torch.flatten(x_emb_hat, start_dim=1),
                    torch.flatten(x_emb, start_dim=1)
                )
            # print('-' * 89)
            # print(x_emb_hat[0])
            # print(x_emb[0, :])
            # print('-' * 89)
        self.metric_['mse'].append(self.mse_.compute())

        return testing_loss

    def encode_data(self, dataloader: DataLoader, max_samples: int = -1) -> torch.Tensor:
        """ Encode data from DataLoader.
        """
        counter = 0
        encodings = []
        self.eval()
        with torch.no_grad():
            for x_num, x_cat in tqdm(dataloader):
                # x_num = rearrange(x_num, 'r c -> c r')
                x_cat = rearrange(x_cat, 'r c -> c r')
                x_num = x_num.to(self.compute_device_)
                x_cat = x_cat.to(self.compute_device_)
                self(x_num, x_cat)
                batch_encoding = self.get_encodings()
                counter += batch_encoding.shape[0]
                if counter > max_samples and max_samples > 0:
                    break
                encodings.append(batch_encoding)
                counter += 1
            encodings = torch.cat(encodings)

        return encodings

    def fit(self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            n_epochs: int) -> dict:
        """ Fit auto-encoder to data coming from `train_loader`, evaluating
            using data from `test_loader`.
        """
        n_params = sum(p.numel() for p in self.parameters())
        print("n_params = ", n_params)
        stop_flag = GracefulExiter()

        training_log = {
            'epoch_wall_time': [],
            'train_loss': [],
            'test_loss': [],
            'reconstruction_error': []}

        logline = "{epoch:+04d},"
        logline += "{time:+04.4f},"
        logline += "{loss:+04.4e},"
        logline += "{test_loss:+04.4e},"
        logline += "{mse:+04.4e},"

        optimizer = torch.optim.Adam(self.parameters(), lr=1E-3)
        for epoch in range(0, n_epochs):
            # train/evaluate
            tic = time.time()

            train_loss = self.train_single_epoch(train_loader, optimizer, n_params)
            test_loss = self.eval_single_epoch(test_loader, n_params)
            toc = time.time()

            # create logging message
            logline = f"{epoch:d}".zfill(5) + ","
            logline += f"{(toc - tic):4.4f}".zfill(5) + " seconds,"
            logline += f"{train_loss:4.4e}".zfill(5) + ","
            logline += f"{test_loss:4.4e}".zfill(5) + ","
            logline += f"{self.metric_['mse'][epoch]:4.4e}".zfill(5) + ","
            # remove trailing comma
            logline = logline[:-1]
            # log = logline.format(epoch=epoch+1,
            #                     time=(toc - tic),
            #                     loss=train_loss,
            #                     test_loss=test_loss,
            #                     test_mse=self.get_test_mse(epoch),
            #                     test_mcp=self.get_test_mcp(epoch) * 100)
            print(logline)
            # update training log
            training_log['epoch_wall_time'].append(toc - tic)
            training_log['train_loss'].append(train_loss)
            training_log['test_loss'].append(test_loss)

            if stop_flag.exit():
                print('-' * 89)
                print('Exiting from training early')
                print('-' * 89)
                break
                # training_log['reconstruction_error'].append(self.get_test_mse(epoch))

        return training_log

    def evaluate(self, val_loader: DataLoader):
        """ Fit auto-encoder to data coming from `train_loader`, evaluating
            using data from `test_loader`.
        """
        n_params = sum(p.numel() for p in self.parameters())
        _ = self.eval_single_epoch(val_loader, n_params)

        return self.metric_['mse'][0]
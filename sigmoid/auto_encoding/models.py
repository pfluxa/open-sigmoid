import time

from typing import Dict, List, NamedTuple, Optional, Tuple, Any

from networkx.algorithms import regular
from networkx.algorithms.shortest_paths.dense import reconstruct_path
from tqdm import tqdm

from einops import rearrange

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import Adam, SGD
from torcheval.metrics import MeanSquaredError

from sigmoid.nn.utils import GracefulExiter


class AutoEmbedderWrapper(torch.nn.Module):
    """ Auto-encoder that trains itself on mixed types of data.
    """
    def __init__(self, config: Dict[str, Any]):

        super(AutoEmbedderWrapper, self).__init__()

        self.config_ = config
        self.model_ = nn.Module()
        self.reconstruction_loss_ = torch.nn.MSELoss(reduction='mean')
        self.metric_ = {}
        self.metric_['num'] = []
        self.metric_['cat'] = []
        self.num_metric_ = MeanSquaredError()
        self.cat_metric_ = MeanSquaredError()
        self.seed_ = 42
        self.compute_device_ = torch.device

        self.model_ = nn.Module()

        self.n_calls_ = 0

    def set_embedding_model(self, model: nn.Module):

        self.model_ = model

    def set_device(self, device: torch.device):
        """ Sets the device to run training/inference.

            @param device: torch.device
                instance of torch.device (ex: torch.device('cpu'))
        """
        self.compute_device_ = device
        self.model_.to(self.compute_device_)
        self.num_metric_ = self.num_metric_.to(self.compute_device_)
        self.cat_metric_ = self.cat_metric_.to(self.compute_device_)

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

        n_num = self.model_.d_num_
        n_cat = self.model_.d_cat_
        
        return n_num + n_cat

    def get_encodings(self) -> torch.Tensor:
        """ Returns encodings of last mini-batch pass.
        """
        return self.code_value_

    def get_embedded_input(self,  transposed: bool = False) -> torch.Tensor:
        """ Returns input embeddings of last mini-batch pass.
        """
        n = self.last_target_num_
        c = self.last_target_cat_
        z = torch.cat([n, c], dim=1)
        
        return z

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor):
        """ Overloads forward method.
        """
        idx_cat = self.n_calls_ % self.model_.n_cat_
        idx_num = self.n_calls_ % self.model_.n_num_
        self.n_calls_ = self.n_calls_ + 1

        x_num, x_cat, lt_num, lt_cat, ct, sec_l = self.model_(
            x_cat, x_cont,
            mask_idx_num=idx_num,
            mask_idx_cat=idx_cat,
        )
        
        self.last_target_num_ = lt_num
        self.last_target_cat_ = lt_cat

        self.code_value_ = ct

        return x_num, x_cat, sec_l

    def train_single_epoch(self,
                           dataloader: DataLoader,
                           optimizer: torch.optim.Optimizer,
                           n_params):
        """ Train auto-encoder for a single epoch.

            @param
        """
        self.model_.train(True)
        epoch_loss = 0.0
        for x_num, x_cat in tqdm(dataloader):
            x_num = rearrange(x_num, 'r c -> c r')
            # x_cat = rearrange(x_cat, 'r c -> c r')
            x_num = x_num.to(self.compute_device_)
            x_cat = x_cat.to(self.compute_device_)

            optimizer.zero_grad()
            e_hat_num, e_hat_cat, sec_loss = self(x_num, x_cat)
            e_num = self.last_target_num_
            e_cat = self.last_target_cat_

            reconstruction_loss = self.reconstruction_loss_(e_hat_num, e_num)
            reconstruction_loss += self.reconstruction_loss_(e_hat_cat, e_cat)
            # regularization_loss = sum(p.abs().sum() for p in self.parameters()) / n_params
            loss = reconstruction_loss #+ sec_loss
            # update running (training) loss
            epoch_loss += loss.item()
            # adjust learning weights
            loss.backward()
            # update uptimizer
            optimizer.step()
        self.model_.train(False)

        return epoch_loss

    def eval_single_epoch(self, dataloader: DataLoader, n_params: int):
        """ Runs model in validation data.
        """
        testing_loss = 0.0
        self.num_metric_.reset()
        self.cat_metric_.reset()

        with torch.no_grad():
            for x_num, x_cat in tqdm(dataloader):
                x_num = rearrange(x_num, 'r c -> c r')
                # x_cat = rearrange(x_cat, 'r c -> c r')
                x_num = x_num.to(self.compute_device_)
                x_cat = x_cat.to(self.compute_device_)
                # run autoencoder on batch
                # x_emb_hat, kld_loss, h_loss = self(x_num, x_cat)
                e_hat_num, e_hat_cat, sec_loss = self(x_num, x_cat)
                e_num = self.last_target_num_
                e_cat = self.last_target_cat_

                reconstruction_loss = self.reconstruction_loss_(e_hat_num, e_num)
                reconstruction_loss += self.reconstruction_loss_(e_hat_cat, e_cat)
                # regularization_loss = sum(p.abs().sum() for p in self.parameters()) / n_params
                # print(f"mse loss = {reconstruction_loss:4.4E}")
                # print(f"sec loss = {sec_loss:4.4E}")
                # print(f"regularization loss = {regularization_loss:4.4E}")
                loss = reconstruction_loss #+ sec_loss
                # update running (testing) loss
                testing_loss += loss.item()
                # compute metrics
                self.num_metric_.update(
                    torch.flatten(e_hat_num, start_dim=1),
                    torch.flatten(e_num, start_dim=1)
                )
                self.cat_metric_.update(
                    torch.flatten(e_hat_cat, start_dim=1),
                    torch.flatten(e_cat, start_dim=1)
                )
            # print('-' * 89)
            # print(x_emb_hat[0])
            # print(x_emb[0, :])
            # print('-' * 89)
        self.metric_['num'].append(self.num_metric_.compute())
        self.metric_['cat'].append(self.cat_metric_.compute())

        return testing_loss

    def evaluate(self, dataloader: DataLoader):
        """ Runs model in validation data.
        """
        self.num_metric_.reset()
        self.cat_metric_.reset()
        with torch.no_grad():
            for x_num, x_cat in tqdm(dataloader):
                x_num = rearrange(x_num, 'r c -> c r')
                # x_cat = rearrange(x_cat, 'r c -> c r')
                x_num = x_num.to(self.compute_device_)
                x_cat = x_cat.to(self.compute_device_)
                # run autoencoder on batch
                # x_emb_hat, kld_loss, h_loss = self(x_num, x_cat)
                e_hat_num, e_hat_cat, _ = self(x_num, x_cat)
                e_num = self.last_target_num_
                e_cat = self.last_target_cat_
                self.num_metric_.update(
                    torch.flatten(e_hat_num, start_dim=1),
                    torch.flatten(e_num, start_dim=1)
                )
                self.cat_metric_.update(
                    torch.flatten(e_hat_cat, start_dim=1),
                    torch.flatten(e_cat, start_dim=1)
                )
        mse_loss = self.num_metric_.compute() + self.cat_metric_.compute() 
        mse_loss = torch.sqrt(mse_loss) / 2.0

        return mse_loss

    def encode_data(self, dataloader: DataLoader, max_samples: int = -1) -> torch.Tensor:
        """ Encode data from DataLoader.
        """
        counter = 0
        encodings = []
        self.eval()
        with torch.no_grad():
            for x_num, x_cat in tqdm(dataloader):
                x_num = rearrange(x_num, 'r c -> c r')
                # x_cat = rearrange(x_cat, 'r c -> c r')
                x_num = x_num.to(self.compute_device_)
                x_cat = x_cat.to(self.compute_device_)
                self(x_num, x_cat)
                batch_encoding = self.get_encodings()
                counter += x_num.shape[1]
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

        # optimizer = Adam(self.parameters(), lr=1E-3) # capturable=True)
        optimizer = SGD(self.parameters(), lr=1E-3, momentum=1e-5) # capturable=True)
        lr_scheduler = ExponentialLR(optimizer, 0.99)
        # lr_scheduler = ReduceLROnPlateau(optimizer,
        #     mode='min',
        #     factor=0.1,
        #     threshold=1E-2,
        #     threshold_mode='abs',
        #     cooldown=2,
        #     patience=3)
        for epoch in range(0, n_epochs):
            # train/evaluate
            tic = time.time()

            train_loss = self.train_single_epoch(train_loader, optimizer, n_params)
            test_loss = self.eval_single_epoch(test_loader, n_params)
            lr_scheduler.step()
            toc = time.time()

            # create logging message
            logline = f"{epoch:d}".zfill(5) + ", "
            logline += "time = " + f"{(toc - tic):4.4f}".zfill(5) + " seconds, "
            logline += "train loss = " + f"{train_loss:4.4e}".zfill(5) + ", "
            logline += "test loss = " + f"{test_loss:4.4e}".zfill(5) + ", "
            logline += "num mse = " + f"{self.metric_['num'][epoch]:4.4e}".zfill(5) + ", "
            logline += "cat mse = " + f"{self.metric_['num'][epoch]:4.4e}".zfill(5) + ", "
            logline += "LR = " + f"{lr_scheduler.get_last_lr()[-1]:4.4e}".zfill(5)
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

    
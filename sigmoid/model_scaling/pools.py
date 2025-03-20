import copy
import math

import torch

from einops import rearrange

from sigmoid.preprocessing.dataloaders import torch_DataLoader
from sigmoid.nn.utils import GracefulExiter

class StochasticPool(torch.nn.Module):
    """ Implementation of a switch MoE.
    """

    @staticmethod
    def dummy(x):
        return x

    @staticmethod
    def as_flat_tensor(x):
        return x.squeeze()

    @staticmethod
    def onehot_to_ordinal(x):
        y = torch.argmax(x, dim=-1)
        return y

    @staticmethod
    def logits_to_ordinal(x):
        x = torch.nn.functional.softmax(x, dim=-1)
        y = torch.argmax(x, dim=-1)
        return y

    def __init__(self, task):
        """ Initializes MoE.
        """
        super(StochasticPool, self).__init__()

        self.autoencoder_ = torch.nn.Module

        self.ne_ = -1
        self.skills_ = torch.nn.ModuleList
        self.s_loss_ = None
        self.skill_metric_ = []

        self.switch_ = torch.nn.Module
        self.routes_ = torch.Tensor
        self.mask_ = torch.Tensor
        self.routing_ = torch.Tensor
        self.switch_loss_ = torch.nn.Module
        self.switch_metrics_ = {}
        self.balancing_ = torch.Tensor
        self.task_ = task

        self.tmax_ = 3.0
        self.tmin_ = 0.1
        self.temp_ = self.tmax_
        self.stop_cooling_ = False
        self.entropy_ = -1.0
        self.curr_skill_ = 0
        self.curr_skill_val_ = 0

        self.device_id_ = torch.device

        self.target_postproc_ = StochasticPool.as_flat_tensor
        if self.task_ == 'classification':
            self.target_postproc_ = StochasticPool.onehot_to_ordinal

        self.prediction_postproc_ = StochasticPool.as_flat_tensor
        if self.task_ == 'classification':
            self.prediction_postproc_ = StochasticPool.onehot_to_ordinal

        self.cooldown_scale_ = None

    def cool_down(self, step):

        k = -math.log(self.tmax_ / self.tmin_)
        new_temp = self.tmax_ * math.exp(k * step / self.cooldown_scale_)
        if new_temp < self.tmin_:
            self.temp_ = self.tmin_
            return False
        else:
            self.temp_ = new_temp
            return True

    def set_cooldown_scale(self, dt: int):
        self.cooldown_scale_ = dt

    def set_device(self, device_id: torch.device):
        """ Set device to run training/inference

            @param device: torch.device
                instance of torch.device (ex: torch.device('cpu'))
        """
        self.device_id_ = device_id

    def set_skills(self, skill_model, n_skills, skill_loss):
        """ Set specialist skills.
        """
        self.ne_ = n_skills
        self.s_loss_ = skill_loss
        self.s_loss_.to(self.device_id_)

        skill_model = skill_model.to(self.device_id_)
        self.skills_ = self._clone_module_list(skill_model, n_skills)

    def set_skill_metric(self, metric, **metric_kwargs):
        """ Sets metric to measure skill performance.

            @param metric: a not-yet-instanciated metric from torcheval.
        """
        self.skill_metric_ = []
        for _ in range(self.ne_):
            m = metric(**metric_kwargs)
            m.to(self.device_id_)
            self.skill_metric_.append(m)

    def set_switch(self, switch_model: torch.nn.Module):
        """ Sets switch.
        """
        self.switch_ = switch_model
        for p in self.switch_.parameters():
            p.requires_grad = False
        # unfreeze very last layer
        for name, param in self.switch_.model_.named_parameters():
            if 'output' in name:
                param.requires_grad = True

        self.switch_.to(self.device_id_)

    def freeze_switch(self):
        for p in self.switch_.parameters():
            p.requires_grad = False

    def set_autoencoder(self, ae_model: torch.nn.Module):
        """ Sets auto-encoder.
        """
        self.autoencoder_ = ae_model
        for name, param in self.autoencoder_.named_parameters():
            print(name)
            param.requires_grad = False
        self.autoencoder_.model_.to(self.device_id_)

    def set_switch_balancing(self, class_weights):
        """ Sets routing balancing scheme.
        """
        w = torch.tensor(class_weights,
                         dtype=torch.float32,
                         device=self.device_id_)
        w = w / w.sum()
        self.balancing_ = w
        self.switch_loss_ = torch.nn.CrossEntropyLoss() #weight=w)
        self.switch_loss_.to(self.device_id_)

    def forward(self, x, y) -> torch.Tensor:
        """ Performs forward pass.

            The switching is performed on a per-batch basis, so that
            different "rows" in the batch get routed to different skills.
        """
        x_num, x_cat = x
        self.autoencoder_(x_num, x_cat)
        codecs = self.autoencoder_.get_encodings()
        embeddings = self.autoencoder_.get_embedded_input()
        logits = self.switch_(codecs)
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        labels = torch.nn.functional.gumbel_softmax(logprobs, tau=self.temp_, hard=True)

        y_list = []
        y_hat_list = []
        invalid_idx = []
        for sidx in range(0, self.ne_):
            bool_msk = labels[:, sidx] > 0.99
            if bool_msk.sum() == 0:
                invalid_idx.append(sidx)
                continue
            msk = torch.where(bool_msk)[0].long()
            e_in = torch.flatten(embeddings[msk, :], start_dim=1)
            y_hat = self.skills_[sidx](e_in)

            y_list.append(y[msk, :])
            y_hat_list.append(y_hat)

        return y_hat_list, y_list, invalid_idx

    def get_routes(self):
        """ Returns routes to experts.
        """
        return self.routes_

    def train_single_epoch(self,
                           epoch: int,
                           data_loader: torch_DataLoader,
                           optimizer: torch.optim.Optimizer):
        self.train()
        epoch_loss = 0.0

        for x_num, x_cat, y in data_loader:
            # x_num = rearrange(x_num, 'r c -> c r')
            x_cat = rearrange(x_cat, 'r c -> c r')
            x_num = x_num.to(self.device_id_)
            x_cat = x_cat.to(self.device_id_)
            y = y.to(self.device_id_)

            optimizer.zero_grad()

            y_hat_list, y_list, bad_idx = self((x_num, x_cat), y)

            sidx = 0
            loss = 0.0
            for i in range(0, self.ne_):
                if i not in bad_idx:
                    loss += self.s_loss_(y_hat_list[sidx], y_list[sidx])
                    sidx += 1
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if not self.stop_cooling_:
            cooldown_success = self.cool_down(epoch)
            if not cooldown_success:
                self.freeze_switch()
                self.stop_cooling_ = True

        return epoch_loss

    def eval_single_epoch(self, data_loader: torch_DataLoader):
        """ Runs model in validation data.
        """

        for metric in self.skill_metric_:
            metric.reset()

        self.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for x_num, x_cat, y in data_loader:
                # x_num = rearrange(x_num, 'r c -> c r')
                x_cat = rearrange(x_cat, 'r c -> c r')
                x_num = x_num.to(self.device_id_)
                x_cat = x_cat.to(self.device_id_)

                y = y.to(self.device_id_)

                y_hat_list, y_list, bad_idx = self((x_num, x_cat), y)

                sidx = 0
                loss = 0.0
                for i in range(0, self.ne_):
                    if i not in bad_idx:
                        loss += self.s_loss_(y_hat_list[sidx], y_list[sidx])
                        sidx += 1
                eval_loss += loss.item()

                sidx = 0
                for i in range(0, self.ne_):
                    if i not in bad_idx:
                        target = self.target_postproc_(y_list[sidx])
                        pred = self.prediction_postproc_(y_hat_list[sidx])
                        self.skill_metric_[sidx].update(
                            pred, target
                        )
                        sidx += 1

            metric_vals = []
            for metric in self.skill_metric_:
                metric_vals.append(metric.compute())

        return eval_loss, metric_vals

    def fit(self, train_loader, val_loader, n_epochs: int):
        """ Trains model on dataset.
        """
        stop_flag = GracefulExiter()
        optimizer = torch.optim.Adam(self.parameters(), lr=1E-4)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1E-5)

        self.curr_skill_ = 0
        self.curr_skill_val_ = 0
        for epoch in range(n_epochs):
            epoch_loss = self.train_single_epoch(epoch, train_loader, optimizer)
            eval_loss, m_values = self.eval_single_epoch(val_loader)
            println = "{:+04d} {:4.5e} {:4.5e} {:4.5e}\n"
            print(println.format(epoch + 1, epoch_loss, eval_loss, self.temp_))
            println = ""
            for sidx in range(len(self.skill_metric_)):
                println += f"  skill_{sidx}: "
                println += f"{m_values[sidx]} \n"
            print(println)

            if stop_flag.exit():
                print('-' * 89)
                print('Exiting from training early')
                print('-' * 89)
                break

    def evaluate(self, data_loader) -> list:
        """ Runs model on evaluation dataset.
        """
        self.eval()
        gt = []
        preds = []
        with torch.no_grad():
            for x_num, x_cat, y in data_loader:
                # x_num = rearrange(x_num, 'r c -> c r')
                x_cat = rearrange(x_cat, 'r c -> c r')
                x_num = x_num.to(self.device_id_)
                x_cat = x_cat.to(self.device_id_)
                y = y.to(self.device_id_)
                y_hat_list, y_list, bad_idx = self((x_num, x_cat), y)
                sidx = 0
                for i in range(0, self.ne_):
                    if i not in bad_idx:
                        target = self.target_postproc_(y_list[sidx])
                        pred = self.prediction_postproc_(y_hat_list[sidx])
                        preds.append(pred)
                        gt.append(target)
                        print(len(preds), len(gt))
                        sidx += 1
                # routes.append(labels)

        predictions = torch.cat(preds).cpu().numpy()
        ground_truth = torch.cat(gt).cpu().numpy()
        # skill_route = torch.cat(routes).cpu().numpy()

        return [ground_truth, predictions] #, skill_route, ]

    @staticmethod
    def _clone_module_list(module, n_clones: int):
        """
        ## Clone Module

        Make a `nn.ModuleList` with clones of a given module
        """
        return torch.nn.ModuleList(
            [copy.deepcopy(module) for _ in range(n_clones)])

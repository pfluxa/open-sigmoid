# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

from sigmoid.preprocessing.dataloaders import torch_DataLoader


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = 1
        # instantiate experts
        self.experts = nn.ModuleList() #[for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def set_experts(self, expert, expert_in, expert_out):
        
        for i in range(self.num_experts):
            self.experts.append(expert(expert_in, expert_out))

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss
    
class MoePool(torch.nn.Module):
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
        super(MoePool, self).__init__()

        self.autoencoder_ = torch.nn.Module
        self.moe_ = None
        self.ne_ = -1
        self.skills_ = torch.nn.ModuleList
        self.s_loss_ = None
        self.skill_metric_ = []

        self.task_ = task

        self.device_id_ = torch.device
        
        self.target_postproc_ = MoePool.as_flat_tensor 
        if self.task_ == 'classification':
            self.target_postproc_ = MoePool.onehot_to_ordinal
        
        self.prediction_postproc_ = MoePool.as_flat_tensor 
        if self.task_ == 'classification':
            self.prediction_postproc_ = MoePool.onehot_to_ordinal

    def set_device(self, device_id: torch.device):
        """ Set device to run training/inference

            @param device: torch.device
                instance of torch.device (ex: torch.device('cpu'))
        """
        self.device_id_ = device_id

    def set_skills(self, n_skills, skill_loss, skill, skill_in, skill_out):
        """ Set specialist skills.
        """
        self.ne_ = n_skills
        self.s_loss_ = skill_loss
        self.s_loss_.to(self.device_id_)

        self.moe_ = MoE(skill_in, skill_out, n_skills, 256)
        self.moe_.set_experts(skill, skill_in, skill_out)
        self.moe_.to(self.device_id_)

    def set_skill_metric(self, metric, **metric_kwargs):
        """ Sets metric to measure skill performance.

            @param metric: a not-yet-instanciated metric from torcheval.
        """
        self.skill_metric_ = []
        m = metric(**metric_kwargs)
        m.to(self.device_id_)
        self.skill_metric_.append(m)

    def set_autoencoder(self, ae_model: torch.nn.Module):
        """ Sets auto-encoder.
        """
        self.autoencoder_ = ae_model
        self.autoencoder_.toggle_decoder()
        for name, param in self.autoencoder_.named_parameters():
            param.requires_grad = False
        self.autoencoder_.to(self.device_id_)

    def forward(self, x) -> torch.Tensor:
        """ Performs forward pass.

            The switching is performed on a per-batch basis, so that
            different "rows" in the batch get routed to different skills.
        """
        x_num, x_cat = x
        self.autoencoder_(x_num, x_cat)
        embeddings = self.autoencoder_.get_embeddings()
        y_hat, moe_loss = self.moe_(embeddings)
        
        return y_hat, moe_loss

    def train_single_epoch(self,
                           epoch: int,
                           data_loader: torch_DataLoader,
                           optimizer: torch.optim.Optimizer):
        self.train()
        epoch_loss = 0.0
        
        for x_num, x_cat, y in data_loader:
            x_num = x_num.to(self.device_id_)
            x_cat = x_cat.to(self.device_id_)
            y = y.to(self.device_id_)
             
            optimizer.zero_grad()
            
            y_hat, moe_loss = self((x_num, x_cat))
            expert_loss = self.s_loss_(y_hat, y)
            loss = moe_loss + expert_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

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
            
                x_num = x_num.to(self.device_id_)
                x_cat = x_cat.to(self.device_id_)
                
                y = y.to(self.device_id_)
                
                y_hat, moe_loss = self((x_num, x_cat))
                expert_loss = self.s_loss_(y_hat, y)
                eval_loss = moe_loss + expert_loss
            
                target = self.target_postproc_(y)
                pred = self.prediction_postproc_(y_hat)
                self.skill_metric_[0].update(
                    pred.long(), target.long()
                )
        metric_vals = []
        for metric in self.skill_metric_:
            metric_vals.append(metric.compute())

        return eval_loss, metric_vals

    def fit(self, train_loader, val_loader, n_epochs: int):
        """ Trains model on dataset.
        """
        # optimizer = torch.optim.Adam(self.parameters(), lr=1E-4)
        optimizer = torch.optim.SGD(self.parameters(), lr=1E-5)

        self.curr_skill_ = 0
        self.curr_skill_val_ = 0
        for epoch in range(n_epochs):
            epoch_loss = self.train_single_epoch(epoch, train_loader, optimizer)
            eval_loss, m_values = self.eval_single_epoch(val_loader)
            println = "{:+04d} {:4.5e} {:4.5e} {:4.5e}\n"
            print(println.format(epoch + 1, epoch_loss, eval_loss, -1.0))
            println = ""
            for sidx in range(len(self.skill_metric_)):
                println += f"  skill_{sidx}: "
                println += f"{m_values[sidx]} \n"
            print(println)
        
    def evaluate(self, data_loader) -> list:
        """ Runs model on evaluation dataset.
        """
        gt = []
        preds = []
        with torch.no_grad():
            for x_num, x_cat, y in data_loader:
                x_num = x_num.to(self.device_id_)
                x_cat = x_cat.to(self.device_id_)
                y = y.to(self.device_id_)
                y_hat, _ = self((x_num, x_cat))
                target = self.target_postproc_(y)
                pred = self.prediction_postproc_(y_hat)
                preds.append(pred)
                gt.append(target)

        predictions = torch.cat(preds).cpu().numpy()
        ground_truth = torch.cat(gt).cpu().numpy()

        return [ground_truth, predictions]
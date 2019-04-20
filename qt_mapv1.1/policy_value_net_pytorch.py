# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, chip_size, max_depth):
        super(Net, self).__init__()
        self.layers = 20
        self.chip_size = chip_size
        self.max_depth = max_depth
        self.batch_norm = nn.BatchNorm2d(1)
        self.circuit_in1 = BatchLinear(max_depth, chip_size, (4 * chip_size) // 3, F.tanh)
        self.circuit_in2 = BatchLinear(max_depth, (4 * chip_size) // 3, 2 * chip_size, F.tanh)
        self.circuit_in3 = BatchLinear(2 * chip_size, max_depth, (4 * max_depth) // 3, F.tanh)
        self.circuit_in4 = BatchLinear(2 * chip_size, (4 * max_depth) // 3, max_depth, F.tanh)
        self.circuit_in5 = BatchLinear(2 * chip_size, max_depth, max_depth // 2, F.tanh)
        self.circuit_in6 = BatchLinear(2 * chip_size, max_depth // 2, max_depth // 4, F.tanh)
        self.circuit_in7 = BatchLinear(2 * chip_size, max_depth // 4, 5, F.tanh)
        self.circuit_in8 = BatchLinear(2 * chip_size, 5, 1, F.tanh)
        self.E_in1 = BatchLinear(1, chip_size, (4 * chip_size) // 3, F.tanh)
        self.E_in2 = BatchLinear(1, (4 * chip_size) // 3, 2 * chip_size, F.tanh)
        self.residual_layers = []
        for i in range(5):
            self.residual_layers.append(BatchLinearNoBatch(1, 2 * chip_size, 2 * chip_size))
        self.act = BatchLinearNoBatch(1, 2 * chip_size, (chip_size - 1) * chip_size // 2 + 1)
        self.val = BatchLinearNoBatch(1, 2 * chip_size, 1)

    def forward(self, state):
        if len(state.size()) == 3:
            state_input = torch.randn([1, 2, self.max_depth, self.chip_size])
            state_input[0] = state.data
        else:
            state_input = torch.randn([state.size(0), 2, self.max_depth, self.chip_size])
            state_input = state.data
        state_input1 = torch.randn([state_input.size(0), 1, self.max_depth, self.chip_size])
        state_input2 = torch.randn([state_input.size(0), 1, 1, self.chip_size])
        state_input1[:, 0, :, :] = copy.deepcopy(state_input[:, 0, :, :])
        state_input2[:, 0, 0, :] = copy.deepcopy(state_input[:, 1, 0, :])
        state_input1 = Variable(state_input1)
        state_input2 = Variable(state_input2)
        x = self.circuit_in1.forward(state_input1, state_input.size(0), 1, False)
        x = self.circuit_in2.forward(x, state_input.size(0), 1, False)
        x = self.circuit_in3.forward(x, state_input.size(0), 1, True)
        x = self.circuit_in4.forward(x, state_input.size(0), 1, False)
        x = self.circuit_in5.forward(x, state_input.size(0), 1, False)
        x = self.circuit_in6.forward(x, state_input.size(0), 1, False)
        x = self.circuit_in7.forward(x, state_input.size(0), 1, False)
        x = self.circuit_in8.forward(x, state_input.size(0), 1, False)
        y = self.E_in1.forward(state_input2, state_input.size(0), 1, False)
        y = self.E_in2.forward(y, state_input.size(0), 1, False)
        x = x.permute(0, 1, 3, 2) + y
        for i in range(5):
            x = F.tanh(self.batch_norm(self.residual_layers[i].forward(x, state_input.size(0), 1, False)) + x)
        x_act = self.act.forward(x, state_input.size(0), 1, False)
        x_act = x_act[:, 0, 0, :]
        x_act = F.log_softmax(x_act)
        x_val = self.val.forward(x, state_input.size(0), 1, False)
        x_val = 0.5 * F.tanh(x_val) + 0.5
        return x_act, x_val

class BatchLinear(object):
    def __init__(self, height, width_in, width_out, stimulate):
        self.height = height
        self.width_in = width_in
        self.width_out = width_out
        self.fc = nn.Linear(width_in, width_out)
        self.batch_norm = nn.BatchNorm2d(1)
        self.stimulate = stimulate

    def forward(self, x, batch, channel, permute):
        for i in range(batch):
            if permute:
                if i == 0:
                    batch_x = ((self.fc(x[i, 0].permute(1, 0))).unsqueeze(0)).unsqueeze(0)
                else:
                    batch_x = torch.cat([batch_x, ((self.fc(x[i, 0].permute(1, 0))).unsqueeze(0)).unsqueeze(0)], 0)
            else:
                if i == 0:
                    batch_x = ((self.fc(x[i, 0])).unsqueeze(0)).unsqueeze(0)
                else:
                    batch_x = torch.cat([batch_x, ((self.fc(x[i, 0])).unsqueeze(0)).unsqueeze(0)], 0)
        batch_x = self.batch_norm(batch_x)
        batch_x = self.stimulate(batch_x)
        return batch_x

class BatchLinearNoBatch(object):
    def __init__(self, height, width_in, width_out):
        self.height = height
        self.width_in = width_in
        self.width_out = width_out
        self.fc = nn.Linear(width_in, width_out)
    
    def forward(self, x, batch, channel, permute):
        for i in range(batch):
            if permute:
                if i == 0:
                    batch_x = ((self.fc(x[i, 0].permute(1, 0))).unsqueeze(0)).unsqueeze(0)
                else:
                    batch_x = torch.cat([batch_x, ((self.fc(x[i, 0].permute(1, 0))).unsqueeze(0)).unsqueeze(0)], 0)
            else:
                if i == 0:
                    batch_x = ((self.fc(x[i, 0])).unsqueeze(0)).unsqueeze(0)
                else:
                    batch_x = torch.cat([batch_x, ((self.fc(x[i, 0])).unsqueeze(0)).unsqueeze(0)], 0)
        return batch_x


class PolicyValueNet():
    """policy-value network """
    def __init__(self, chip_size, max_depth,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(chip_size, max_depth).cuda()
        else:
            self.policy_value_net = Net(chip_size, max_depth)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = []
        for i in board.available:
            if i is not None:
                legal_positions.append(board.loc2mov(i))
        current_state = board.current_state()
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data.numpy()
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        mse_loss = torch.nn.MSELoss()
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.data[0], entropy.data[0]
        #for pytorch version >= 0.5 please use the following line instead.
        #return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)

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
        kernel_size1 = 11
        kernel_size2 = 7
        kernel_size3 = 5
        self.chip_size = chip_size
        self.max_depth = max_depth
        # circuit input 
        #self.circuit_in1 = nn.Conv2d(1, 4, kernel_size1, padding = kernel_size1 // 2)
        #self.circuit_in2 = nn.Conv2d(4, 64, kernel_size1, padding = kernel_size1 // 2)
        #self.circuit_in3 = nn.Conv2d(64, 128, kernel_size1, padding = kernel_size1 // 2)
        self.circuit_in1 = nn.Linear(chip_size, chip_size)
        # E input
        #self.E_in1 = nn.Conv1d(1, self.max_depth // 4, kernel_size2, padding = kernel_size2 // 2)
        #self.E_in2 = nn.Conv1d(self.max_depth // 4, self.max_depth // 2, kernel_size2, padding = kernel_size2 // 2)
        #self.E_in3 = nn.Conv1d(self.max_depth // 2, self.max_depth, kernel_size2, padding = kernel_size2 // 2)
        self.E_in1 = nn.Linear(chip_size, max_depth)
        self.E_in2 = nn.Linear(1, chip_size)
        # common layers
        #self.conv1 = nn.Conv2d(128, 256, kernel_size3, padding = kernel_size3 // 2)
        #self.conv11 = nn.Conv2d(256, 256, kernel_size3, padding = kernel_size3 // 2)
        #self.conv2 = nn.Conv2d(256, 128, kernel_size3, padding = kernel_size3 // 2)
        #self.conv22 = nn.Conv2d(128, 128, kernel_size3, padding = kernel_size3 // 2)
        #self.conv3 = nn.Conv2d(128, 64, kernel_size3, padding = kernel_size3 // 2)
        #self.conv33 = nn.Conv2d(64, 64, kernel_size3, padding = kernel_size3 // 2)
        self.conv1 = nn.Linear(chip_size, 2 * chip_size)
        self.conv11 = nn.Linear(2 * chip_size, 2 * chip_size)
        self.conv2 = nn.Linear(2 * chip_size, 4 * chip_size)
        self.conv22 = nn.Linear(4 * chip_size, 4 * chip_size)
        self.conv3 = nn.Linear(4 * chip_size, 8 * chip_size)
        self.conv33 = nn.Linear(8 * chip_size, 8 * chip_size)
        # action policy layers
        #self.act_conv1 = nn.Conv2d(64, 4, kernel_size=1)
        #self.act_fc1 = nn.Linear(4 * chip_size * max_depth,
        #                         chip_size * (chip_size - 1))
        self.act_fc1 = nn.Linear(8 * chip_size, (chip_size * (chip_size - 1)) // 2 + 1)
        self.act_fc2 = nn.Linear(max_depth, max_depth)
        self.act_fc3 = nn.Linear(max_depth, 1)
        # state value layers
        #self.val_conv1 = nn.Conv2d(64, 2, kernel_size=1)
        #self.val_fc1 = nn.Linear(2* chip_size * max_depth, 16)
        #self.val_fc2 = nn.Linear(16, 1)
        self.val_fc1 = nn.Linear(8 * chip_size, chip_size)
        self.val_fc2 = nn.Linear(chip_size, 1)
        self.val_fc3 = nn.Linear(max_depth, max_depth)
        self.val_fc4 = nn.Linear(max_depth, 1)

    def forward(self, state):
        if len(state.size()) == 3:
            state_input = torch.randn([1, 2, self.max_depth, self.chip_size])
            state_input[0] = state.data
        else:
            state_input = torch.randn([state.size(0), 2, self.max_depth, self.chip_size])
            state_input = state.data
        state_input1 = torch.randn([state_input.size(0), self.max_depth, self.chip_size])
        state_input2 = torch.randn([state_input.size(0), 1, self.chip_size])
        state_input1 = copy.deepcopy(state_input[:, 0, :, :])
        state_input2[:, 0, :] = copy.deepcopy(state_input[:, 1, 0, :])
        state_input1 = Variable(state_input1)
        state_input2 = Variable(state_input2)
        for i in range(len(state_input)):
            # input layers
            # circuit
            #x = self.circuit_in1(state_input1)
            #x = self.circuit_in2(x)
            #x = self.circuit_in3(x)
            x = self.circuit_in1(state_input1[i])   
            # E
            #y = self.E_in1(state_input2)
            #y = self.E_in2(y)
            #y = self.E_in3(y)
            y = self.E_in1(state_input2[i])
            y = y.permute(1, 0)
            y = self.E_in2(y)
            # add
            #y = y.view(self.max_depth, self.chip_size)
            #for i in range(x.size(1)):
            #    x[0, i, :, :] = x[0, i, :, :] + y[:, :]
            x = x + y
            # common layers
            x = F.relu(self.conv1(x))
            for ii in range(10):
                x = F.relu(self.conv11(x) + x)
            x = F.relu(self.conv2(x))
            for ii in range(10):
                x = F.relu(self.conv22(x) + x)
            x = F.relu(self.conv3(x))
            for ii in range(10):
                x = F.relu(self.conv33(x) + x)
            # action policy layers
            #x_act = F.relu(self.act_conv1(x))
            #x_act = x_act.view(-1, 4 * self.chip_size * self.max_depth)
            #x_act = F.log_softmax(self.act_fc1(x_act))
            x_act = self.act_fc1(x)
            x_act = x_act.permute(1, 0)
            x_act = self.act_fc2(x_act)
            x_act = self.act_fc3(x_act)
            x_act = F.log_softmax(F.tanh(x_act.permute(1, 0)))###########has to be changed
            # state value layers
            #x_val = F.relu(self.val_conv1(x))
            #x_val = x_val.view(-1, 2 * self.chip_size * self.max_depth)
            x_val = self.val_fc1(x)
            x_val = self.val_fc2(x_val)
            x_val = x_val.permute(1, 0)
            x_val = F.relu(self.val_fc3(x_val))
            x_val = 1 / 2 + F.tanh(self.val_fc4(x_val)) / 2
            if i == 0:
                total_x_act = x_act
                total_x_val = x_val
            else:
                total_x_act = torch.cat((total_x_act, x_act), 0)
                total_x_val = torch.cat((total_x_val, x_val), 0)
        return total_x_act, total_x_val


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
        value = value.data[0][0]
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

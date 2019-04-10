import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import net_generate as ng
import copy
from torch.autograd import Variable
from rule import Chip, Compile
from torch import tensor
 
class Net(nn.Module):
    def __init__(self, chip_size, max_depth):
        super(Net, self).__init__()
        self.chip_size = chip_size
        self.max_depth = max_depth
        self.circuit_in1 = BatchLinear(chip_size, (4 * chip_size) // 3, F.tanh)
        self.circuit_in2 = BatchLinear((4 * chip_size) // 3, 2 * chip_size, F.tanh)
        self.circuit_in3 = BatchLinear(max_depth, (4 * max_depth) // 3, F.tanh)
        self.circuit_in4 = BatchLinear((4 * max_depth) // 3, max_depth, F.tanh)
        self.circuit_in5 = BatchLinear(max_depth, max_depth // 2, F.tanh)
        self.circuit_in6 = BatchLinear(max_depth // 2, max_depth // 4, F.tanh)
        self.circuit_in7 = BatchLinear(max_depth // 2, max_depth // 4, F.tanh)
        self.circuit_in8 = BatchLinear(max_depth // 4, 5, F.tanh)
        self.circuit_in8 = BatchLinear(5, 1, F.tanh)
        self.E_in1 = BatchLinear(chip_size, (4 * chip_size) // 3, F.tanh)
        self.E_in2 = BatchLinear((4 * chip_size) // 3, 2 * chip_size, F.tanh)
        self.residual_layers = []
        for i in range(5):
            self.residual_layers.append(BatchLinearNoBatch(2 * chip_size, 2 * chip_size, F.tanh))
        self.out = BatchLinear(2 * chip_size, (chip_size - 1) * chip_size // 2 + 1)

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
        # circuit input

        return 
 
class BatchLinear(object):
    def __init__(self, height, width, stimulate):
        self.height = height
        self.width = width
        self.fc = nn.Linear(height, width)
        self.batch_norm = nn.BatchNorm2d(1)
        self.stimulate = stimulate

    def forward(self, x, batch, channel, permute):
        batch_x = Variable(torch.randn([batch, channel, self.height, self.width]))
        for i in range(len(batch)):
            if permute:
                batch_x[i, 0] = self.fc(x[i, 0].permute(1, 0))
            else:
                batch_x[i, 0] = self.fc(x[i, 0])
        batch_x = self.batch_norm(batch_x)
        batch_x = self.stimulate(batch_x)
        return batch_x

class BatchLinearNoBatch(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.fc = nn.Linear(height, width)
    
    def forward(self, x, batch, channel, permute):
        batch_x = Variable(torch.randn([self.batch, self.channel, self.height, self.width]))
        for i in range(len(batch)):
            if permute:
                batch_x[i, 0] = self.fc(x[i, 0].permute(1, 0))
            else:
                batch_x[i, 0] = self.fc(x[i, 0])
        return batch_x

class DQN(object):
    def __init__(self, chip_size, max_depth, model_file=None, use_gpu=False):
        self.gamma = 0.5  # reward discount
        self.target_replace_iter = 100  # target update frequency
        self.l2_const = 1e-4
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.eval_net, self.target_net = Net(chip_size, max_depth).cuda(), Net(chip_size, max_depth).cuda()
        else:
            self.eval_net, self.target_net = Net(chip_size, max_depth), Net(chip_size, max_depth)
        if model_file:
            net_params = torch.load(model_file)
            self.eval_net.load_state_dict(net_params)
            self.target_net.load_state_dict(net_params)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), weight_decay=self.l2_const)

    def fn(self, state, net):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = []
        for i in state.available:
            if i is not None:
                legal_positions.append(state.loc2mov(i))
        current_state = state.current_state()
        if self.use_gpu:
            act_probs = net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = act_probs.data.cpu().numpy().flatten()
        else:
            act_probs = net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = act_probs.data.numpy().flatten()
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs 

    def batch_fn(self, state_batch, net):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            act_probs = net(state_batch)
            act_probs = act_probs.data.cpu().numpy()
            return act_probs
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            act_probs = net(state_batch)
            act_probs = act_probs.data.numpy()
            return act_probs

    def train_step(self, state_batch, action_batch, reward_batch, next_state_batch, lr):
        self.target_net.load_state_dict(self.eval_net.state_dict())
        mse_loss = torch.nn.MSELoss()
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            action_batch = Variable(torch.FloatTensor(action_batch).cuda())
            reward_batch = Variable(torch.FloatTensor(reward_batch).cuda())
            next_state_batch = Variable(torch.FloatTensor(next_state_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            action_batch = Variable(torch.FloatTensor(action_batch))
            reward_batch = Variable(torch.FloatTensor(reward_batch))
            next_state_batch = Variable(torch.FloatTensor(next_state_batch))
        a_b = torch.zeros([1, len(action_batch.data)])
        a_b = a_b.long()
        a_b[0, :] = action_batch.data
        a_b = a_b.permute(1, 0)
        q_eval = self.eval_net(state_batch).gather(1, Variable(a_b))
        q_next = self.target_net(next_state_batch).detach()
        q_target = reward_batch + self.gamma * q_next.max(1)[0].float()
        loss = mse_loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def get_param(self):
        net_params = self.eval_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_param()  # get model params
        torch.save(net_params, model_file)
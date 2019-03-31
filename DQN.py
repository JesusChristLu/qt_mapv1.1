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
        kernel_size1 = 11
        kernel_size2 = 7
        kernel_size3 = 5
        self.chip_size = chip_size
        self.max_depth = max_depth
        self.circuit_in1 = nn.Linear(chip_size, chip_size)
        self.E_in1 = nn.Linear(chip_size, max_depth)
        self.E_in2 = nn.Linear(1, chip_size)
        self.conv1 = nn.Linear(chip_size, 2 * chip_size)
        self.conv11 = nn.Linear(2 * chip_size, 2 * chip_size)
        self.conv2 = nn.Linear(2 * chip_size, 4 * chip_size)
        self.conv22 = nn.Linear(4 * chip_size, 4 * chip_size)
        self.conv3 = nn.Linear(4 * chip_size, 8 * chip_size)
        self.conv33 = nn.Linear(8 * chip_size, 8 * chip_size)
        self.act_fc1 = nn.Linear(8 * chip_size, (chip_size * (chip_size - 1)) // 2 + 1)
        self.act_fc2 = nn.Linear(max_depth, max_depth)
        self.act_fc3 = nn.Linear(max_depth, 1)

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
            x_act = F.log_softmax(F.relu(x_act.permute(1, 0)))###########has to be changed
            if i == 0:
                total_x_act = x_act
            else:
                total_x_act = torch.cat((total_x_act, x_act), 0)
        return total_x_act
 
class DQN(object):
    def __init__(self, chip_size, max_depth, model_file=None, use_gpu=False):
        self.gamma = 0.9  # reward discount
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
        a_b = torch.zeros([1, 5])
        a_b = a_b.long()
        a_b[0, :] = action_batch.data
        a_b = a_b.permute(1, 0)
        q_eval = self.eval_net(state_batch).gather(1, Variable(a_b))
        q_next = self.target_net(next_state_batch).detach()
        q_target = reward_batch + self.gamma * q_next.max(1)[1].float()
        loss = mse_loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
# -*- coding: utf-8 -*-

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from rule import Chip, Compile
from naive import Naive
from copy import deepcopy
import net_generate as ng
#from naive import MCTSPlayer as MCTS_Pure !!!!!!!!!!!!!!
from MTCScompiler import MCTSPlayer
# from policy_value_net_keras import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras
from DQN import DQN
from DQN_player import DQN_player
from net_generate import Circuit_Generate
from torch.autograd import Variable
from rule import Chip, Compile
from train import count_remain_gate
import torch
import torch.nn

class DQN_train():
    def __init__(self, newnet, logical_number, circuit, count_remain_gate, init_model=None):
        # params of the board and the game
        self.chip = Chip(newnet)
        self.circuit = circuit
        self.compile = Compile(self.chip, count_remain_gate)
        self.epsilon = 0.9  # greedy policy
        self.target_replace_iter = 100  # target update frequency
        self.memory_capacity = 500
        #self.memory_capacity = 500###################
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.batch_size = 100  # mini-batch size for training
        #self.batch_size = 80##########################
        self.memory = deque(maxlen=self.memory_capacity)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.check_freq = 50
        self.game_batch_num = 50
        #self.game_batch_num = 100#########################
        self.kl_targ = 0.02
        if init_model:
            # start training from an initial policy-value net
            self.dqn = DQN(self.chip.chip_size, self.chip.max_depth, init_model)
        else:
            # start training from a new policy-value net
            self.dqn = DQN(self.chip.chip_size, self.chip.max_depth)
        self.dqn_player = DQN_player(self.chip, self.dqn, is_selfplay=1)

    def collect_selfplay_data(self, whole_depth, logical_number, n_games = 1):
        """collect self-play data for training"""
        for i in range(n_games):
            play_data, total_swap_number, time = self.compile.start_dqn_self_play(self.dqn_player, whole_depth,
                                                     logical_number, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            data = []
            for i in range(self.episode_len - 1):
                data.append(play_data[i] + (play_data[i + 1][0],))
            self.memory.extend(data)
        return total_swap_number, time

    def get_param(self):
        net_params = self.dqn.eval_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_param()  # get model params
        torch.save(net_params, model_file)

    def learn(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.memory, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        action_batch = [data[1] for data in mini_batch]
        complete_batch = [data[2] for data in mini_batch]
        next_s_batch = [data[3] for data in mini_batch]
        old_probs = self.dqn.batch_fn(state_batch, self.dqn.eval_net)
        for i in range(self.epochs):
            loss = self.dqn.train_step(
                        state_batch,
                        action_batch,
                        complete_batch,
                        next_s_batch,
                        self.learn_rate*self.lr_multiplier)
            new_probs = self.dqn.batch_fn(state_batch, self.dqn.eval_net)
            kl = np.mean(np.sum(old_probs * (
                    np.log(np.abs(old_probs) + 1e-10) - np.log(np.abs(new_probs) + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        return loss

    def run(self, whole_depth, logical_number):
        '''run the training'''
        #init_model = '.\\model\\dqn_net3_3.model'
        #self.dqn = DQN(self.chip.chip_size, self.chip.max_depth, init_model)
        #self.dqn_player = DQN_player(self.chip, self.dqn, is_selfplay=1)
        losses = np.zeros([self.game_batch_num])
        times = np.zeros([self.game_batch_num])
        swaps = np.zeros([self.game_batch_num])
        for i in range(self.game_batch_num):
            whole_depth = random.randint(9, 51)
            logical_number = random.randint(2, 11)
            swap, time = self.collect_selfplay_data(whole_depth, logical_number, self.play_batch_size)
            print('batch i:{}, episode_len:{}'.format(i + 1, self.episode_len))
            if len(self.memory) > self.batch_size:
                loss = self.learn()
                loss = loss.data.numpy()[0]
                losses[i] = loss
                times[i] = time / (whole_depth * logical_number)
                swaps[i] = swap / (whole_depth * logical_number)
                print(loss, swap)
        write_dat('.\\loss\\dqn_4x4', losses, 'a', 'f')
        write_dat('.\\data\\dqn_swap_4x4', swaps, 'a', 'f')
        write_dat('.\\data\\dqn_time_4x4', times, 'a', 'f')
        self.dqn.save_model('.\\model\\dqn_net4x4.model')

def read_dat(name, type):
    with open(name, 'r') as file_to_read:
        data = []
        while True:
            lines = file_to_read.readline() 
            if not lines:
                break
            if type == 'i':
                data.append([int(i) for i in lines.split()])
            else:
                data.append([float(i) for i in lines.split()])
    file_to_read.close()
    return data

def write_dat(name, data, mode, type):
    with open(name, mode) as f1:
        for i in range(len(data)):
            if type == 'i':
                f1.write(str(int(data[i])))
            else:
                f1.write(str(float(data[i])))
            f1.write(' ')
        f1.write('\n')
        f1.close()
        

if __name__ == '__main__':
    whole_depth = 30
    logical_bit_number = 5
    net = ng.Net_Generate(N = 0, name = '.\\chips\\chip_4x4.txt', is_read = True)
    circuit = ng.Circuit_Generate(whole_depth, logical_bit_number)
    #net.draw_graph()
    #print(circuit.q)
    dqn_train = DQN_train(net, logical_bit_number, circuit, count_remain_gate)
    dqn_train.run(whole_depth, logical_bit_number)
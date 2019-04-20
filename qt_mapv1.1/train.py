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

class TrainPipeline():
    def __init__(self, newnet, logical_number, circuit, count_remain_gate, init_model=None):
        # params of the board and the game
        self.chip = Chip(newnet)
        self.circuit = circuit
        self.count_remain_gate = count_remain_gate
        self.compile = Compile(self.chip, self.count_remain_gate)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 0.1  # the temperature param
        self.n_playout = len(self.chip.available) * 10 # num of simulations for each move 
        self.c_puct = 1
        self.buffer_size = 10000
        self.batch_size = 500  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 2
        self.game_batch_num = 10
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.chip.chip_size, self.chip.max_depth,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.chip.chip_size, self.chip.max_depth)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, 
                                      self.count_remain_gate,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def collect_selfplay_data(self, whole_depth, logical_number, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            play_data, total_swap_number, time = self.compile.start_self_play(self.mcts_player, whole_depth,
                                                     logical_number, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)
        return total_swap_number, time

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        complete_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    complete_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(complete_batch) - old_v.flatten()) /
                             np.var(np.array(complete_batch)))
        explained_var_new = (1 -
                             np.var(np.array(complete_batch) - new_v.flatten()) /
                             np.var(np.array(complete_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        naive_player = Naive(self.chip, self.compile)
#       naive_player = MCTS_Pure(c_puct=5,
#                                     n_playout=self.pure_mcts_playout_num)
#
#       dqn_player = ??
        win_cnt = []
        cnt = 0
        circuit_storage = deepcopy(self.circuit)
        for i in range(n_games):
            self.circuit = deepcopy(circuit_storage)
            swap_pvn = self.compile.start_play(current_mcts_player, self.circuit)
            self.circuit = deepcopy(circuit_storage)
            swap_nai = self.compile.start_play(naive_player, self.circuit)
#           tiem_dqn = self.chip.start_play(dqn_player,
#                                          pure_mcts_player,
#                                          start_player=i % 2,
#                                          is_shown=0)
            win_cnt.append(([swap_pvn, swap_nai]))
        for i in win_cnt:
            if i[0] < i[1]:
                cnt += 1
        return cnt / len(win_cnt)

    def run(self, whole_depth, logical_number):
        """run the training pipeline"""
        
        init_model = '.\\model\\MTCS.model'
        self.policy_value_net = PolicyValueNet(self.chip.chip_size, self.chip.max_depth,
                                                   model_file=init_model)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      count_remain_gate,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
        
        losses = []
        entropys = []
        times = []
        swaps = []
        try:
            for i in range(self.game_batch_num):
                whole_depth = random.randint(10, 51)
                logical_number = random.randint(2, 11)
                swap, time = self.collect_selfplay_data(whole_depth, logical_number, self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    losses.append(loss)
                    entropys.append(entropy)
                    times.append(time / (whole_depth * logical_number))
                    swaps.append(swap / (whole_depth * logical_number))
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    #win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('.\\model\\MTCS.model')
            write_dat('.\\loss\\MTCS_loss_4x4', losses, 'a', 'f')
            write_dat('.\\loss\\MTCS_entropy_4x4', entropys, 'a', 'f')
            write_dat('.\\data\\MTCS_swap_4x4', swaps, 'a', 'f')
            write_dat('.\\data\\MTCS_time_4x4', times, 'a', 'f')
            '''
            losses = np.array(losses)
            entropys = np.array(entropys)
            x = np.arange(0, self.game_batch_num, 1) 
            plt.figure(figsize=(5,3.5))
            plt.plot(x, losses, 'r', marker='o', label='loss')
            plt.plot(x, entropys, 'g', marker='^', label='entropy')
            plt.legend(loc='best') 
            plt.xlabel('game num')
            plt.ylabel(' ')
            plt.title('loss function and entropy')
            #plt.savefig('G:/YDPIC/example.png',dpi=1200) 
            plt.show()   
            '''           
        except KeyboardInterrupt:
            print('\n\rquit')

def count_remain_gate(q1, q2):
    control = []
    remain = 0
    for i in range(len(q1)):
        if q1[i] != q2[i]:
            if q1[i] and (abs(q1[i]) not in control):
                control.append(abs(q1[i]))
                remain += 1
            elif q2[i] and (abs(q2[i]) not in control):
                control.append(abs(q2[i]))
                remain += 1
    return remain

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
    chip_size = 9 #3x3
    logical_bit_number = 5
    depth = 30
    circuit = ng.Circuit_Generate(depth, logical_bit_number)
    net = ng.Net_Generate(N = 0, name = '.\\chips\\chip_4x4.txt', is_read = True)
    training_pipeline = TrainPipeline(net, logical_bit_number, circuit, count_remain_gate)
    training_pipeline.run(depth, logical_bit_number)

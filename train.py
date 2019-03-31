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
    def __init__(self, newnet, logical_number, circuit, init_model=None):
        # params of the board and the game
        self.chip = Chip(newnet)
        self.circuit = circuit
        self.compile = Compile(self.chip, logical_number)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 10  # num of simulations for each move!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.c_puct = 5
        self.buffer_size = 100
        self.batch_size = 5  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 2
        self.game_batch_num = 150
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
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def collect_selfplay_data(self, whole_depth, logical_number, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            play_data = self.compile.start_self_play(self.mcts_player, whole_depth,
                                                     logical_number, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)

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
#        naive_player = MCTS_Pure(c_puct=5,
#                                     n_playout=self.pure_mcts_playout_num)
#
#        dqn_player = ??
        win_cnt = []
        cnt = 0
        circuit_storage = deepcopy(self.circuit)
        for i in range(n_games):
            self.circuit = deepcopy(circuit_storage)
            swap_pvn = self.compile.start_play(current_mcts_player, self.circuit)
            self.circuit = deepcopy(circuit_storage)
            swap_nai = self.compile.start_play(naive_player, self.circuit)
#            tiem_dqn = self.chip.start_play(dqn_player,
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
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(whole_depth, logical_number, self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                '''
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                '''
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    logical_number = 5
    chip_size = 8
    whole_depth = 10
    net = ng.Net_Generate(0, 1, chip_size)
    circuit = ng.Circuit_Generate(whole_depth, logical_number)
    #net.draw_graph()
    #print(circuit.q)
    training_pipeline = TrainPipeline(net, logical_number, circuit)
    training_pipeline.run(whole_depth, logical_number)

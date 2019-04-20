from DQN import DQN
import torch
from torch.autograd import Variable
import numpy as np

class DQN_player:
    def __init__(self, chip, dqn, is_selfplay=0):
        self.chip = chip
        self.dqn = dqn
        self._is_selfplay = is_selfplay
        self.epsilon = 0.9  # greedy policy

    def get_action(self, chip, remain_depth, temp=1, return_prob=1):
        action_value = self.dqn.fn(chip, self.dqn.eval_net)
        actions, values = zip(*action_value)
        values = torch.Tensor(np.array(values))
        if self._is_selfplay:    
            if np.random.uniform() < self.epsilon:
                action = actions[torch.max(values, 0)[1].numpy()[0]]
            else: # random
                action = actions[np.random.randint(0, len(actions))]
        else:
            action = actions[torch.max(values, 0)[1].numpy()[0]]
        return action, values.numpy()

    def reset_player(move):
        return
# -*- coding: utf-8 -*-

import numpy as np
import copy
from copy import deepcopy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._w = 0
        self._u = 0
        self._P = prior_p

    def expand(self, actions, probs):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for i in range(len(actions)):
            if (actions[i] not in self._children):
                self._children[actions[i]] = TreeNode(self, probs[i])

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._w += leaf_value
        self._Q = self._w / self._n_visits

    def update_recursive(self, leaf_value, gamma):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(gamma * leaf_value, gamma)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (0.1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, count_remain_gate, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._gamma = 0.9
        self._lambda = 0.7
        self._discount = 0.9
        self._punish = 0.7
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._action = 0
        self.count_remain_gate = count_remain_gate

    def _playout(self, state, r_d, w_d):
        #max((swap_number - 1), 0)
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        remain_depth = r_d
        whole_depth = w_d
        curr_q = deepcopy(state.state['q'].q[0])
        nothing_had_done = False
        integral = 0
        swap_number = 0
        #deep = 0
        while(1):
            #q = []
            #u = []
            #ucb = []
            #for i in node._children.items():
            #    q.append(i[1]._Q)
            #    u.append(i[1]._u)
            #    ucb.append(i[1]._Q + i[1]._u)
            #print(q)
            #deep += 1
            #print(u)
            #print(ucb)
            small_reward = 0
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            #print(deep, action, node)
            self._action = action
            temp_remain_depth, swap_number = state.do_move(action, remain_depth, nothing_had_done, simulate = True)
            integral = (whole_depth - remain_depth) + (remain_depth - temp_remain_depth) * (self._discount ** swap_number)
            if remain_depth > temp_remain_depth:
                remain_depth = temp_remain_depth
                nothing_had_done = False
            elif False not in (state.state['q'].q[0] == curr_q):
                nothing_had_done = True
            else:
                nothing_had_done = False
                small_reward = self.count_remain_gate(curr_q, state.state['q'].q[0]) / (len([i for i in state.state['E'] if i > 0]) // 2)
            if remain_depth:
                curr_q = deepcopy(state.state['q'].q[0])  
            else:
                break
        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [0, 1]
        # for the current player.
        action_probs, leaf_value_eval = self._policy(state)
        actions, probs = [], []
        #for i in action_probs:
        #    actions.append(i[0])
        #    probs.append(i[1])
        actions, probs = zip(*action_probs)
        # Check for end of game.
        leaf_value_search = (integral + small_reward) / whole_depth 
        if remain_depth:
            node.expand(actions, probs)
        #else:
        #    print('good! you are so fucking lucky!!!!')
        # Update value and visit count of nodes in this traversal.
        leaf_value = self._lambda * leaf_value_eval + (1 - self._lambda) * leaf_value_search
        node.update_recursive(leaf_value * (self._punish ** swap_number), self._gamma)

    def get_move_probs(self, state, remain_depth, whole_depth, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy, remain_depth, whole_depth)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        '''
        if len(self._root._children.items()) == 0 and remain_depth == 0:
            return None, None
        '''
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        print(visits)################################################################
        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)
        
    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function, count_remain_gate,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, count_remain_gate, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, remain_depth, whole_depth, temp=1e-3, return_prob=0):
        sensible_moves = board.available
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.chip_size * (board.chip_size - 1) // 2 + 1)
        acts, probs = self.mcts.get_move_probs(board, remain_depth, whole_depth, temp)
        '''
        if acts == None:
            if return_prob:
                return None, None
            else:
                return None
        '''
        move_probs[list(acts)] = probs
        if self._is_selfplay:
            # add Dirichlet Noise for exploration (needed for
            # self-play training)
            move = np.random.choice(
                acts,
                p=0.9*probs + 0.1*np.random.dirichlet(0.3*np.ones(len(probs)))
            )
            # update the root node and reuse the search tree
            self.mcts.update_with_move(move)
        else:
            # with the default temp=1e-3, it is almost equivalent
            # to choosing the move with the highest prob
            move = np.random.choice(acts, p=probs)
            # reset the root node
            self.mcts.update_with_move(-1)
#            location = board.move_to_location(move)
#            print("AI move: %d,%d\n" % (location[0], location[1]))

        if return_prob:
            return move, move_probs
        else:
            return move

    def __str__(self):
        return "MCTS {}".format(self.player)

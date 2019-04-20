import numpy as np
import random
import net_generate
import networkx as nx
import naive as nv
from copy import deepcopy

class Chip(object):
    def __init__(self, newnet):
        self.g = newnet
        self.connection = newnet.get_net_structure()
        self.chip_size = newnet.get_net_size()
        self.is_dig = newnet.get_net_kind()
        self.max_depth = 100
        self.available = self.__find_available__()

    def init_chip(self, logical_bit_number, circuit):
        E = np.zeros(self.chip_size,dtype = int)
        location = random.sample(range(0, self.chip_size), logical_bit_number)
        E[location] = list(range(1, logical_bit_number + 1))
        self.E = E
        self.state = {}
        self.state['E'] = self.E
        self.state['q'] = circuit
        self.last_move = -1

    def __find_available__(self):
        available = []
        for i in range(self.chip_size):
            for j in range(i + 1, self.chip_size):
                if self.connection[i, j]:
                    available.append((i, j))
                else:
                    available.append(None)
        available.append((-1, -1))
        return available
    
    def mov2loc(self, move):
        return self.available[move]

    def loc2mov(self, loc):
        return self.available.index(loc)

    def current_state(self):
        curr_state = np.zeros([2, self.max_depth, self.chip_size])
        if len(self.state['q'].q) > self.max_depth:
            curr_state[0, 0 : self.max_depth, 
                       0 : len(self.state['q'].q[0])] = self.state['q'].q[0 : self.max_depth, :]
        elif len(self.state['q'].q): 
            curr_state[0, 0 : len(self.state['q'].q), 
                       0 : len(self.state['q'].q[0])] = self.state['q'].q
        self.state['E'] = self.E
        curr_state[1, 0, :] = self.state['E']
        return curr_state

    def do_move(self, move, remain_depth, nothing_had_done = False, simulate = False):
        swap_number = 0
        if (self.last_move != move) or (not nothing_had_done):
            loc = self.mov2loc(move)
            if -1 not in loc:
                self.E[loc[0]], self.E[loc[1]] = self.E[loc[1]], self.E[loc[0]]
                swap_number += 1
            #if not simulate:
            #    print('the swap', loc)
        curr_state = self.current_state()
        #if not simulate:
        #    print('before swap: ', curr_state[0,0])
        for i in range(self.chip_size):# maybe can be changed to logical number
            if curr_state[0, 0, i]:
                for j in range(i + 1, self.chip_size):# maybe can be changed to logical number
                    if not(curr_state[0, 0, i] + curr_state[0, 0, j]):
                        qj = np.where(self.E == j + 1)
                        qi = np.where(self.E == i + 1)
                        qi = int(qi[0])
                        qj = int(qj[0])
                        #print(curr_state[0, 0, i], curr_state[0, 0, j], self.connection[qi, qj])
                        if (self.last_move == move) and nothing_had_done:                     
                            path = nx.shortest_path(self.g.return_g,source = qi,target = qj)
                            for ii in range(len(path) - 1):
                                self.E[path[ii]], self.E[path[ii + 1]] = self.E[path[ii + 1]], self.E[path[ii]]
                                loc = []
                                loc.append(path[ii])
                                loc.append(path[ii + 1])
                                loc = tuple(loc)
                                #if not simulate:
                                #    print('the swap', loc)
                                swap_number += 1
                            qj = np.where(self.E == j + 1)
                            qi = np.where(self.E == i + 1)
                            qi = int(qi[0])
                            qj = int(qj[0])
                        if self.connection[qi, qj]:
                            curr_state[0, 0, int(i)] = curr_state[0, 0, int(j)] = 0
                            self.state['q'].q[0] = curr_state[0, 0, 0 : len(self.state['q'].q[0])]
                        break
        #if not simulate:
        #    print('after swap: ', curr_state[0,0])
        if False not in (np.zeros(self.chip_size) == curr_state[0, 0, :]):
                self.state['q'].q = self.state['q'].q[1 :]
                remain_depth -= 1
        self.last_move = move
        return remain_depth, swap_number

    def compute_layers_distance(self, gamma = 0.03):
        curr_state = self.current_state()
        distance = 0
        for n in range(len(curr_state[0])):
            for i in range(self.chip_size):
                if curr_state[0, n, i]:
                    for j in range(self.chip_size):
                        if not(curr_state[0, n, i] + curr_state[0, n, j]):
                            qi = np.where(self.E == i + 1)
                            qj = np.where(self.E == j + 1)
                            distance += (gamma ** n) * nx.shortest_path_length(self.g.return_g, int(qi[0]), int(qj[0]))
                            break
        return distance


class Compile(object):
    def __init__(self, chip, logical_number, count_remain_gate):
        self.chip = chip
        self.logical_number = logical_number
        self.discount = 0.9
        self.miu = 0.9
        self.count_remain_gate = count_remain_gate

    def start_play(self, compiler, circuit):
        self.circuit = deepcopy(circuit)
        self.chip.init_chip(self.logical_number, self.circuit)
        self.whole_depth = len(self.circuit.q)
        self.remain_depth = self.whole_depth
        time = 0
        total_swap_number = 0
        nothing_had_done = False
        curr_q = deepcopy(self.chip.state['q'].q[0])
        #print('time: ', time, ', remain depth: ', self.remain_depth)
        while True:
            #print('the before map: ', self.chip.state['E'])
            move = compiler.get_action(self.chip, self.remain_depth)
            temp_remain_depth, swap_number = self.chip.do_move(move, self.remain_depth, nothing_had_done)
            if self.remain_depth > temp_remain_depth:
                self.remain_depth = temp_remain_depth
                nothing_had_done = False
            elif False not in (self.chip.state['q'].q[0] == curr_q):
                nothing_had_done = True
            else:
                nothing_had_done = False
            if self.remain_depth:
                curr_q = deepcopy(self.chip.state['q'].q[0])
            total_swap_number += swap_number
            time += 1
            #print('the after map: ', self.chip.state['E'])
            #print('time: ', time, ', remain depth: ', self.remain_depth, 'swap number: ', swap_number)
            if not self.remain_depth:
                #print('total_swap_number: ', total_swap_number)
                return time, total_swap_number

    def start_self_play(self, player, whole_depth, logical_number, is_shown=0, temp=1e-3):
        #max((swap_number - 1), 0)
        circuit = net_generate.Circuit_Generate(whole_depth, logical_number)
        self.chip.init_chip(self.logical_number, circuit)
        self.whole_depth = len(circuit.q)
        self.remain_depth = self.whole_depth
        states, mcts_probs, completness, integrals = [], [], [], []
        time = 0
        total_swap_number = 0
        nothing_had_done = False
        curr_q = deepcopy(self.chip.state['q'].q[0])
        #print('time: ', time, ', remain depth: ', self.remain_depth)
        while True:
            #print('the before map: ', self.chip.state['E'])
            move, move_probs = player.get_action(self.chip,
                                                 self.remain_depth,
                                                 self.whole_depth,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.chip.current_state())
            mcts_probs.append(move_probs)
            # perform a move
            swap_number = 0
            temp_remain_depth, swap_number = self.chip.do_move(move, self.remain_depth, nothing_had_done)
            integral = (self.whole_depth - self.remain_depth + 
                       (self.remain_depth - temp_remain_depth) * (self.discount ** swap_number))
            if self.remain_depth > temp_remain_depth: 
                self.remain_depth = temp_remain_depth
                nothing_had_done = False
                small_reward = 0
            elif False not in (self.chip.state['q'].q[0] == curr_q):
                nothing_had_done = True
                small_reward = 0
            else:
                nothing_had_done = False
                small_reward = self.count_remain_gate(self.chip.state['q'].q[0], curr_q) / (self.logical_number // 2)
            if self.remain_depth:
                curr_q = deepcopy(self.chip.state['q'].q[0])   
            completness.append(total_swap_number + min(1, swap_number))
            integrals.append((integral + small_reward) / self.whole_depth)
            time += 1
            total_swap_number += swap_number 
            #print('the after map: ', self.chip.state['E'])
            #print('time: ', time, ', remain depth: ', self.remain_depth, 'swap number: ', swap_number)
            if not self.remain_depth:
                if total_swap_number == 0:
                    total_swap_number = 0.01
                for i in range(time):
                    completness[i] = completness[i] / total_swap_number
                    completness[i] = (1 - self.miu) * completness[i] + self.miu * integrals[i]
                #print('total_swap_number: ', total_swap_number)
                player.reset_player()
                return (zip(states, mcts_probs, completness), total_swap_number, time)

    def start_dqn_self_play(self, player, whole_depth, logical_number, is_shown=0, temp=1e-3):
        #max((swap_number - 1), 0)
        circuit = net_generate.Circuit_Generate(whole_depth, logical_number)
        self.chip.init_chip(self.logical_number, circuit)
        self.whole_depth = len(circuit.q)
        self.remain_depth = self.whole_depth
        states, moves, rewards, next_states = [], [], [], []
        foresee = 0.03
        time = 0
        total_swap_number = 0
        nothing_had_done = False
        curr_q = deepcopy(self.chip.state['q'].q[0])
        #print('time: ', time, ', remain depth: ', self.remain_depth)
        while True:
            #print('the before map: ', self.chip.state['E'])
            move, move_probs = player.get_action(self.chip,
                                                 self.remain_depth,
                                                 temp=temp,
                                                 return_prob=1)
            swap_number = 0
            pre_distance = self.chip.compute_layers_distance(foresee)
            # store the data
            states.append(self.chip.current_state())
            moves.append(move)
            # perform a move
            temp_remain_depth, swap_number = self.chip.do_move(move, self.remain_depth, nothing_had_done)
            next_states.append(self.chip.current_state())
            if self.remain_depth > temp_remain_depth:
                reward = (self.remain_depth - temp_remain_depth) * (0.1 ** max((swap_number - 1), 0))
                rewards.append(reward)
                nothing_had_done = False
                self.remain_depth = temp_remain_depth
            elif False in (self.chip.state['q'].q[0] == curr_q):
                reward = self.count_remain_gate(self.chip.state['q'].q[0], curr_q) / (self.logical_number // 2)
                rewards.append(reward)
                nothing_had_done = False
            else:
                reward = (pre_distance - self.chip.compute_layers_distance(foresee)) / 1000
                rewards.append(reward)  
                nothing_had_done = True
            if self.remain_depth:
                curr_q = deepcopy(self.chip.state['q'].q[0])
            time += 1
            total_swap_number += swap_number
            #print('the after map: ', self.chip.state['E'])
            #print('time: ', time, ', remain depth: ', self.remain_depth, 'swap number: ', swap_number)
            if not self.remain_depth:
                #print('total_swap_number: ', total_swap_number)
                player.reset_player()
                return (zip(states, moves, rewards, next_states), total_swap_number, time)

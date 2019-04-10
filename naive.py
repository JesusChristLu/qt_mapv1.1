from copy import deepcopy
import random
class Naive(object):
    
    def __init__(self, chip, compiler, gamma = 0.01):
        self.chip = chip
        self.compiler = compiler
        self.gamma = gamma

    def list_next_state(self, E):
        next_E = {}
        for i in self.chip.available:
            newE = deepcopy(E)
            if i is not None: 
                if -1 not in i:
                    newE[i[0]], newE[i[1]] = newE[i[1]], newE[i[0]]
                next_E[i] = newE
        return next_E

    def get_action(self, chip, useless):
        newchip = deepcopy(chip)
        next_E = self.list_next_state(newchip.E)
        distance = newchip.compute_layers_distance()
        shortest_E = (-1, -1)
        distances = []
        for i in next_E:
            newchip.E = next_E[i]
            newchip.state['E'] = next_E[i]
            temp_dist = newchip.compute_layers_distance(self.gamma)
            distances.append(temp_dist)
            if temp_dist < distance:
                distance = temp_dist
                shortest_E = i
        return chip.loc2mov(shortest_E)
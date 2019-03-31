import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

class Net_Generate:
    '''
    a model chip generator
    '''
    def __init__(self, is_digraph, is_self_generate, N = 0):
        self.is_DiGraph = is_digraph
        self.is_Self_Generate = is_self_generate
        self.number = N
        self.return_g, self.connect_matrix = self.generate()

    def generate(self):
        if self.is_DiGraph:
            if self.is_Self_Generate:
                G, C = self._generate_self_(self.number, self.is_DiGraph)
            else:
                G, C = self._generate_inner_(self.number, self.is_DiGraph)
        else:
            if self.is_Self_Generate:
                G, C = self._generate_self_(self.number, self.is_DiGraph)
            else:
                G, C = self._generate_inner_(self.number, self.is_DiGraph)
        return G, C

    def _generate_self_(self, N, is_dig):
        C = np.zeros([N, N])
        if is_dig:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for i in range(0, N):
            #print(i)
            p = random.randint(0, 100)
            if p < 5:
                edge_number = 1
            elif p < 60:
                edge_number = 2
            elif p < 80:
                edge_number = 3
            else:
                edge_number = 4
            while edge_number > N - len(range(0, i)) - 1:
                edge_number = edge_number - 1    
            connected = random.sample(range(0, N), edge_number)
            while set(range(0, i + 1)) & set(connected):
                connected = random.sample(range(0, N), edge_number)
            C[i, connected] = 1
            if is_dig:
                C[connected, i] = -1
            else:
                C[connected, i] = 1
            G.add_edges_from([(i, j) for j in connected])
        return G, C

    def _generate_inner_(self, N, is_dig):
        ########################################ini_G=nx.grid_2d_graph(4,3) 
        ini_G=nx.grid_2d_graph(2,2) 
        C = nx.to_numpy_matrix(ini_G)
        if is_dig:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for i in range(len(C)):
            for j in range(i, len(C)):
                if C[i, j]:
                    if is_dig:
                        if random.randint(0,100) > 50:
                            G.add_edges_from([(i, j)])
                            C[j, i] = -1
                        else:
                            G.add_edges_from([(j, i)])
                            C[i, j] = -1
                    else:
                        G.add_edges_from([(i, j)])
        return G, C

    def get_net_size(self):
        return self.number

    def get_net_kind(self):
        return self.is_DiGraph

    def get_net_structure(self):
        return self.connect_matrix

    def get_net(self):
        return return_g

    def draw_graph(self):
        nx.draw(self.return_g, with_labels=True) 
        plt.title('demo chip') 
        plt.axis('off') 
        plt.xticks([]) 
        plt.yticks([]) 
        plt.show()

class Circuit_Generate:
    '''
    a random program generator
    '''
    def __init__(self, dep, number):
        self.log_bit_number = number
        self.depth = dep
        self.q = self.__circuit__()

    def __circuit__(self):
        qq = np.zeros([self.depth, self.log_bit_number])
        while False not in (np.zeros([self.depth, self.log_bit_number]) == qq):
            for i in range(self.depth):
                p =  random.randint(0, 100)
                if p < 70:
                    cnots = random.randint(0, min([5, self.log_bit_number // 2]))
                elif p < 80:
                    cnots = random.randint(0, max([self.log_bit_number // 10, 1]))
                elif p < 90:
                    cnots = random.randint(0, max([self.log_bit_number // 5, 1]))
                else:
                    cnots = random.randint(0, max([self.log_bit_number // 3, 1]))
                cnot_bit = random.sample(range(self.log_bit_number), cnots * 2)
                for ii in range(0, len(cnot_bit) - 1, 2):
                    qq[i, cnot_bit[ii]] = int(ii + 1)
                    qq[i, cnot_bit[ii + 1]] = int(-ii - 1)
        return qq

def run():
    toy_net = Net_Generate(0, 0, 4)
    toy_net.draw_graph()

run()
    
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

class Net_Generate:
    '''
    a model chip generator
    '''
    def __init__(self, N = 0, name = None, is_read = True, is_rand = False,  edge = 1, is_dig = False):
        self.number = N
        self.name = name
        self.is_read = is_read
        self.is_rand = is_rand
        self.edge = edge
        self.is_dig = is_dig
        self.return_g, self.connect_matrix = self.generate()
        self.number = len(self.connect_matrix)

    def generate(self):
        if self.is_read:
            G, C = self.read_net(self.name)
        elif self.is_rand:
            G, C = self._generate_rand_()
        else:
            G, C = self._generate_squre_chip_()
        if self.is_dig:
            G, C = self._change_to_dig_(G, C)
        return G, C

    def _generate_rand_(self):
        C = np.zeros([self.number, self.number])
        G = nx.Graph()
        for i in range(0, self.number):
            #print(i)
            p = random.randint(0, 100)
            if p < 2:
                edge_number = self.edge
            elif p < 60:
                edge_number = self.edge + 1
            elif p < 80:
                edge_number = self.edge + 2
            elif p < 90:
                edge_number = self.edge + 3
            elif p < 95:
                edge_number = self.edge + 4
            elif p < 96:
                edge_number = self.edge + 5
            elif p < 97:
                edge_number = self.edge + 6
            elif p < 98:
                edge_number = self.edge + 7
            else:
                edge_number = self.edge + 8
            while edge_number > self.number - len(range(0, i)) - 1:
                edge_number = edge_number - 1    
            connected = random.sample(range(0, self.number), edge_number)
            while set(range(0, i + 1)) & set(connected):
                connected = random.sample(range(0, self.number), edge_number)
            C[i, connected] = C[connected, i] = 1
            G.add_edges_from([(i, j) for j in connected])
        return G, C

    def _generate_squre_chip_(self):
        if float(int(float(self.number) ** 0.5)) == float(self.number) ** 0.5:
            M = int(self.number ** 0.5)
        else:
            M = int(self.number ** 0.5) - 1
        N = self.number // M
        G = nx.grid_2d_graph(M, N) 
        C = nx.to_numpy_matrix(G)
        G = nx.from_numpy_matrix(C)
        return G, C

    def _change_to_dig_(self, G, C):
        G = nx.DiGraph()
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                if C[i, j]:
                    if random.randint(0,100) > 50:
                        G.add_edges_from([(i, j)])
                        C[j, i] = -1
                    else:
                        G.add_edges_from([(j, i)])
                        C[i, j] = -1
        return G, C


    def read_net(self, name):
        G = nx.Graph()
        with open(name, 'r') as file_to_read:
            C = []
            while True:
                lines = file_to_read.readline() 
                if not lines:
                    break
                C.append([int(i) for i in lines.split()])
        file_to_read.close()
        C = np.matrix(C)
        G = nx.from_numpy_matrix(C)
        return G, C

    def write_net(self, name):
        with open('.\\chips\\'+name+'.txt', 'w') as f1:
            for i in range(len(self.connect_matrix)):
                for j in range(len(self.connect_matrix)):
                    f1.write(str(int(self.connect_matrix[i, j])))
                    f1.write(' ')
                f1.write('\n')
            f1.close()

    def get_net_size(self):
        return self.number

    def get_net_kind(self):
        return self.is_dig

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
    def __init__(self, dep, number, pr = 1):
        self.log_bit_number = number
        self.depth = dep
        self.pr = pr
        self.q = self._circuit_()

    def _circuit_(self):
        qq = np.zeros([self.depth, self.log_bit_number])
        while False not in (np.zeros([self.depth, self.log_bit_number]) == qq):
            for i in range(self.depth):
                p =  random.randint(0, 100)
                if p < 30:#70
                    cnots = random.randint(0, min([5, self.log_bit_number // 2]))
                elif p < 40:#80
                    cnots = random.randint(0, max([self.log_bit_number // 10, 1]))
                elif p < 50:#90
                    cnots = random.randint(0, max([self.log_bit_number // 5, 1]))
                else:
                    cnots = random.randint(0, max([self.log_bit_number // 3, 1]))
                cnot_bit = random.sample(range(self.log_bit_number), cnots * 2)
                for ii in range(0, len(cnot_bit) - 1, 2):
                    qq[i, cnot_bit[ii]] = int(ii + 1)
                    qq[i, cnot_bit[ii + 1]] = int(-ii - 1)
        return qq

def run():
    '''
    nets = []
    for i in range(0, 10):
        nets.append(Net_Generate(N = 20, name = None, 
                    is_read = False, is_rand = True, edge = i + 1, is_dig = False))
    for i in range(0, 10):
        nets[i].write_net('chip_4x5_' + str(i+1))
        nets[i].draw_graph()
    '''

    return 

run()

    
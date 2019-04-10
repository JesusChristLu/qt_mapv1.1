from __future__ import print_function
import pickle
import net_generate as ng
import naive as nv
from rule import Chip
from rule import Compile
from train import count_remain_gate
from policy_value_net_pytorch import PolicyValueNet
from MTCScompiler import MCTSPlayer
import matplotlib.pyplot as plt
import numpy as np
cnames = ['black','blue','gray','green','orange','purple','red','tomato','yellow']

def run():
    '''
    standard net 9 bits, square, edge 1
    standard circuit 30 depth, 5 bits, p? 
    '''

    '''
    net = ng.Net_Generate(N = 0, name = '.\\chips\\chip_4x4.txt', is_read = True)
    #net.draw_graph()
    logical_bit_number = 5
    total_times = []
    total_swaps = []
    for ii in range(10, 80, 10):
        depth = ii
        circuit = ng.Circuit_Generate(depth, logical_bit_number)
        model_file = '.\\model\\MTCS.model'
        try:
            chip = Chip(net)
            compile = Compile(chip, logical_bit_number, count_remain_gate)
            try:
                policy_param = pickle.load(open(model_file, 'rb'))
            except:
                policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
        #best_policy = PolicyValueNet(chip.chip_size, chip.max_depth, policy_param)
        #mcts_player = MCTSPlayer(best_policy.policy_value_fn,
        #                         c_puct=5,
        #                         n_playout=400)  # set larger n_playout for better performance
        #dqn_player = ??

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)
            times, swaps = [], []
            for i in range(30):
                time = 0
                swap = 0
                for epoch in range(5):
                    naive_player = nv.Naive(chip, compile, float(i / 50))
                    t, s = compile.start_play(naive_player, circuit)
                    time += t
                    swap += s
                time /= 5
                swap /= 5
                times.append(time)
                swaps.append(swap)
                print(str(int(i * 2)) + '%')
            times = np.array(times)
            swaps = np.array(swaps)
            write_data('.\\data\\times4x4.txt', times,'a','f')
            write_data('.\\data\\swaps4x4.txt', swaps, 'a', 'f')
        except KeyboardInterrupt:
            print('\n\rquit')
        print('depth: ' + str(depth))
    
    total_swaps = read_data('.\\data\\swaps4x4.txt', 'f')
    total_swaps = np.array(total_swaps)
    total_times = read_data('.\\data\\times4x4.txt', 'f')
    total_times = np.array(total_times)
    x = np.arange(0, 0.6, 1 / 50) 
    fig = plt.figure(figsize=(5,3.5))
    ax1 = fig.add_subplot(121)
    ax1.set_title('the relationship between gamma and swap')
    ax1.set_xlabel('gamma')       
    ax1.set_ylabel(' ')
    for i in range(0, 7):
        plt.plot(x, total_swaps[i], cnames[i], marker='^', label=('depth = ' + str((i + 1) * 10)))       
    plt.legend(loc='best') 
    ax2 = fig.add_subplot(122)
    ax2.set_title('the relationship between gamma and compilation time')
    ax2.set_xlabel('gamma')  
    ax2.set_ylabel(' ') 
    for i in range(0, 7):
        plt.plot(x, total_times[i], cnames[i], marker='o', label=('depth = ' + str((i + 1) * 10)))
    plt.legend(loc='best') 
    #plt.title('the relationship between gamma and swap or compilation time')
    #plt.savefig('gamma.jpg',dpi=1200) 
    plt.show()  
    '''
    
    loss3 = read_data('.\\loss\\dqn_3x3_5_30', 'f')
    times = read_data('.\\data\\dqn_time_3x3_5_30', 'f')
    swaps = read_data('.\\data\\dqn_swap_3x3_5_30', 'f')
    t_and_s = []
    t_and_s.append(times)
    t_and_s.append(swaps)
    labels = ['times', 'swaps']
    x = np.arange(len(loss3[0])) 
    fig = plt.figure(figsize=(5,3.5))
    ax1 = fig.add_subplot(121)
    ax1.set_title('loss function')
    ax1.set_xlabel('epoch')       
    ax1.set_ylabel('loss')
    plt.plot(x, loss3[0], cnames[1], marker='^')       
    plt.legend(loc='best') 
    ax2 = fig.add_subplot(122)
    ax2.set_title('time and swap')
    ax2.set_xlabel('epoch')  
    ax2.set_ylabel(' ') 
    for i in range(0, 2):
        plt.plot(x, t_and_s[i][0], cnames[i], marker='o', label=(labels[i]))
    plt.legend(loc='best') 
    #plt.title('the relationship between gamma and swap or compilation time')
    #plt.savefig('gamma.jpg',dpi=1200) 
    plt.show()  
    
    '''
    mtcs_swap = read_data('.\\data\\MTCS_swap_3x3_5_30', 'i')
    mtcs_time = read_data('.\\data\\MTCS_time_3x3_5_30', 'i')
    mtcs_loss = read_data('.\\loss\\MTCS_loss_3x3_5_30', 'f')
    mtcs_entropy = read_data('.\\loss\\MTCS_entropy_3x3_5_30', 'f')
    x = np.arange(200)
    fig = plt.figure(figsize=(5,3.5))
    ax1 = fig.add_subplot(121)
    ax1.set_title('')
    ax1.set_xlabel('epoch')       
    ax1.set_ylabel('swap number')
    plt.plot(x, mtcs_swap[0][0:200], cnames[1], marker='o')       
    plt.legend(loc='best') 
    ax2 = fig.add_subplot(122)
    ax2.set_title('')
    ax2.set_xlabel('epoch')  
    ax2.set_ylabel('compilation time') 
    plt.plot(x, mtcs_swap[0][0:200], cnames[2], marker='o')
    plt.legend(loc='best') 
    #plt.title('the relationship between gamma and swap or compilation time')
    #plt.savefig('gamma.jpg',dpi=1200) 
    plt.show()  
    fig2 = plt.figure(figsize=(5,3.5))
    ax1 = fig2.add_subplot(121)
    ax1.set_title('')
    ax1.set_xlabel('epoch')       
    ax1.set_ylabel('loss')
    plt.plot(x, mtcs_loss[0][0:200], cnames[3], marker='o')       
    plt.legend(loc='best') 
    ax2 = fig2.add_subplot(122)
    ax2.set_title('')
    ax2.set_xlabel('epoch')  
    ax2.set_ylabel('entropy') 
    plt.plot(x, mtcs_entropy[0][0:200], cnames[4], marker='o')
    plt.legend(loc='best') 
    #plt.title('the relationship between gamma and swap or compilation time')
    #plt.savefig('gamma.jpg',dpi=1200) 
    plt.show()  
    '''

def read_data(name, type):
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

def write_data(name, data, mode, type):
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
    run()
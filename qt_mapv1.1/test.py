from __future__ import print_function
import pickle
import net_generate as ng
import naive as nv
from rule import Chip
from rule import Compile
from train import count_remain_gate
from policy_value_net_pytorch import PolicyValueNet
from MTCScompiler import MCTSPlayer
from DQN import DQN
from DQN_player import DQN_player
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
    depth = 30
    total_times = []
    total_swaps = []
    bit_numbers = [6, 8, 10, 12, 14, 16]
    for ii in bit_numbers:
        logical_bit_number = ii
        circuit = ng.Circuit_Generate(depth, logical_bit_number)
        chip = Chip(net)
        compile = Compile(chip, logical_bit_number, count_remain_gate)
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
        write_dat('.\\data\\times4x4_bit.txt', times,'a','f')
        write_dat('.\\data\\swaps4x4_bit.txt', swaps, 'a', 'f')
        print('bit number: ' + str(logical_bit_number))
    
    total_swaps = read_dat('.\\data\\swaps4x4_bit.txt', 'f')
    total_swaps = np.array(total_swaps)
    total_times = read_dat('.\\data\\times4x4_bit.txt', 'f')
    total_times = np.array(total_times)
    x = np.arange(0, 0.6, 1 / 50) 
    fig = plt.figure(figsize=(5,3.5))
    ax1 = fig.add_subplot(121)
    ax1.set_title('the relationship between gamma and swap')
    ax1.set_xlabel('gamma')       
    ax1.set_ylabel(' ')
    for i in range(len(bit_numbers)):
        plt.plot(x, total_swaps[i], cnames[i], marker='^', label=('bit number = ' + str((bit_numbers[i]))))       
    plt.legend(loc='best') 
    ax2 = fig.add_subplot(122)
    ax2.set_title('the relationship between gamma and compilation time')
    ax2.set_xlabel('gamma')  
    ax2.set_ylabel(' ') 
    for i in range(len(bit_numbers)):
        plt.plot(x, total_times[i], cnames[i], marker='o', label=('bit number = ' + str((bit_numbers[i]))))
    plt.legend(loc='best') 
    #plt.title('the relationship between gamma and swap or compilation time')
    #plt.savefig('gamma.jpg',dpi=1200) 
    plt.show()  
    '''
    '''
    loss3 = read_dat('.\\loss\\dqn_4x4', 'f')
    times = read_dat('.\\data\\dqn_time_4x4', 'f')
    swaps = read_dat('.\\data\\dqn_swap_4x4', 'f')
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
    ax2.set_ylabel('*(whole_depth * logical_number)') 
    for i in range(0, 2):
        plt.plot(x, t_and_s[i][0], cnames[i], marker='o', label=(labels[i]))
    plt.legend(loc='best') 
    #plt.title('the relationship between gamma and swap or compilation time')
    #plt.savefig('gamma.jpg',dpi=1200) 
    plt.show()  
    '''
 
    mtcs_swap = read_dat('.\\data\\MTCS_swap_4x4', 'f')
    mtcs_time = read_dat('.\\data\\MTCS_time_4x4', 'f')
    mtcs_loss = read_dat('.\\loss\\MTCS_loss_4x4', 'f')
    mtcs_entropy = read_dat('.\\loss\\MTCS_entropy_4x4', 'f')
    x = np.arange(len(mtcs_swap[0]))
    fig = plt.figure(figsize=(5,3.5))
    ax1 = fig.add_subplot(121)
    ax1.set_title('')
    ax1.set_xlabel('epoch')       
    ax1.set_ylabel('*(whole_depth * logical_number)')
    plt.plot(x, mtcs_swap[0], cnames[1], marker='o')       
    plt.legend(loc='best') 
    ax2 = fig.add_subplot(122)
    ax2.set_title('')
    ax2.set_xlabel('epoch')  
    ax2.set_ylabel('*(whole_depth * logical_number)') 
    plt.plot(x, mtcs_time[0], cnames[2], marker='o')
    plt.legend(loc='best') 
    #plt.title('the relationship between gamma and swap or compilation time')
    #plt.savefig('gamma.jpg',dpi=1200) 
    plt.show()  
    fig2 = plt.figure(figsize=(5,3.5))
    ax1 = fig2.add_subplot(121)
    ax1.set_title('')
    ax1.set_xlabel('epoch')       
    ax1.set_ylabel('loss')
    plt.plot(x, mtcs_loss[0], cnames[3], marker='o')       
    plt.legend(loc='best') 
    ax2 = fig2.add_subplot(122)
    ax2.set_title('')
    ax2.set_xlabel('epoch')  
    ax2.set_ylabel('entropy') 
    plt.plot(x, mtcs_entropy[0], cnames[4], marker='o')
    plt.legend(loc='best') 
    #plt.title('the relationship between gamma and swap or compilation time')
    #plt.savefig('gamma.jpg',dpi=1200) 
    plt.show()  

'''   
    depth = 30
    net = ng.Net_Generate(N = 0, name = '.\\chips\\chip_3x3.txt', is_read = True)
    chip = Chip(net)
    compile = Compile(chip, count_remain_gate)
    mtcs_model = '.\\model\\MTCS.model'
    best_policy = PolicyValueNet(chip.chip_size, chip.max_depth, mtcs_model)
    mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                             count_remain_gate,
                             c_puct=1,
                             n_playout=400)
    dqn_model = '.\\model\\dqn_net3_3.model'
    dqn = DQN(chip.chip_size, chip.max_depth, dqn_model)
    dqn_player = DQN_player(chip, dqn)
    naive_player = nv.Naive(chip, compile, float(0.1))
    dqn_time = []
    dqn_swap = []
    mtcs_time = []
    mtcs_swap = []
    naive_time = []
    naive_swap = []
    for ii in range(8):
        logical_bit_number = ii + 2
        circuit = ng.Circuit_Generate(depth, logical_bit_number)
        t, s = compile.start_play(naive_player, circuit, logical_bit_number)
        naive_time.append(t)
        naive_swap.append(s)
        t, s = compile.start_play(mcts_player, circuit, logical_bit_number)
        mtcs_time.append(t)
        mtcs_swap.append(s)
        t, s = compile.start_play(dqn_player, circuit, logical_bit_number)
        dqn_time.append(t)
        dqn_swap.append(s)   
    x = range(8)
    fig = plt.figure(figsize=(5,3.5))
    ax1 = fig.add_subplot(121)
    ax1.set_title('')
    ax1.set_xlabel('bit number')       
    ax1.set_ylabel('swap')
    plt.plot(x, naive_swap, cnames[1], marker='o', label='naive player')   
    plt.plot(x, mtcs_swap, cnames[2], marker='o', label='mtcs player')  
    plt.plot(x, dqn_swap, cnames[3], marker='o', label='dqn player')      
    plt.legend(loc='best') 
    ax2 = fig.add_subplot(122)
    ax2.set_title('')
    ax2.set_xlabel('bit number')  
    ax2.set_ylabel('time') 
    plt.plot(x, naive_time, cnames[1], marker='o', label='naive player')   
    plt.plot(x, mtcs_time, cnames[2], marker='o', label='mtcs player')  
    plt.plot(x, dqn_time, cnames[3], marker='o', label='dqn player')     
    plt.legend(loc='best') 
    #plt.title('the relationship between gamma and swap or compilation time')
    #plt.savefig('gamma.jpg',dpi=1200) 
    plt.show()
        
    logical_bit_number = 6
    net = ng.Net_Generate(N = 0, name = '.\\chips\\chip_3x3.txt', is_read = True)
    chip = Chip(net)
    compile = Compile(chip, count_remain_gate)
    mtcs_model = '.\\model\\MTCS.model'
    best_policy = PolicyValueNet(chip.chip_size, chip.max_depth, mtcs_model)
    mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                             count_remain_gate,
                             c_puct=1,
                             n_playout=400)
    dqn_model = '.\\model\\dqn_net3_3.model'
    dqn = DQN(chip.chip_size, chip.max_depth, dqn_model)
    dqn_player = DQN_player(chip, dqn)
    naive_player = nv.Naive(chip, compile, float(0.1))
    dqn_time = []
    dqn_swap = []
    mtcs_time = []
    mtcs_swap = []
    naive_time = []
    naive_swap = []
    for ii in range(10):
        depth = (ii + 1) * 10
        circuit = ng.Circuit_Generate(depth, logical_bit_number)
        t, s = compile.start_play(naive_player, circuit, logical_bit_number)
        naive_time.append(t)
        naive_swap.append(s)
        t, s = compile.start_play(mcts_player, circuit, logical_bit_number)
        mtcs_time.append(t)
        mtcs_swap.append(s)
        t, s = compile.start_play(dqn_player, circuit, logical_bit_number)
        dqn_time.append(t)
        dqn_swap.append(s)   
    x = range(10, 101, 10)
    fig = plt.figure(figsize=(5,3.5))
    ax1 = fig.add_subplot(121)
    ax1.set_title('')
    ax1.set_xlabel('depth')       
    ax1.set_ylabel('swap')
    plt.plot(x, naive_swap, cnames[1], marker='o', label='naive player')   
    plt.plot(x, mtcs_swap, cnames[2], marker='o', label='mtcs player')  
    plt.plot(x, dqn_swap, cnames[3], marker='o', label='dqn player')      
    plt.legend(loc='best') 
    ax2 = fig.add_subplot(122)
    ax2.set_title('')
    ax2.set_xlabel('depth')  
    ax2.set_ylabel('time') 
    plt.plot(x, naive_time, cnames[1], marker='o', label='naive player')   
    plt.plot(x, mtcs_time, cnames[2], marker='o', label='mtcs player')  
    plt.plot(x, dqn_time, cnames[3], marker='o', label='dqn player')     
    plt.legend(loc='best') 
    #plt.title('the relationship between gamma and swap or compilation time')
    #plt.savefig('gamma.jpg',dpi=1200) 
    plt.show()    
    return 
'''
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
    run()
from __future__ import print_function
import pickle
import net_generate as ng
import naive as nv
from rule import Chip
from rule import Compile
from policy_value_net_pytorch import PolicyValueNet
from MTCScompiler import MCTSPlayer

def run():
    '''
    net = ng.Net_Generate(1, 1, 6)
    net.draw_graph()
    net = ng.Net_Generate(1, 0, 6)
    net.draw_graph()
    net = ng.Net_Generate(0, 1, 6)
    net.draw_graph()
    '''
    net = ng.Net_Generate(0, 1, 15)
    #net.draw_graph()
    logical_bit_number = 12
    depth = 50
    circuit = ng.Circuit_Generate(depth, logical_bit_number)
    #print(circuit.q)
    model_file = 'best_policy_8_8_5.model'
    try:
        chip = Chip(net)
        compile = Compile(chip, logical_bit_number)

        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
        #best_policy = PolicyValueNet(chip.chip_size, chip.max_depth, policy_param)
        #mcts_player = MCTSPlayer(best_policy.policy_value_fn,
        #                         c_puct=5,
        #                         n_playout=400)  # set larger n_playout for better performance
        naive_player = nv.Naive(chip, compile)
        #dqn_player = ??

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)
        '''
        mcts_time = compile.start_play(mcts_player)
        '''
        naive_time = compile.start_play(naive_player, circuit)
        '''
        dqn_time = compile.start_play(dqn_time)
        '''
    except KeyboardInterrupt:
        print('\n\rquit')

if __name__ == '__main__':
    run()

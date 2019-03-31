# qt_mapv1.1
Three method of mapping between quantum circuit and quantum hardware.
1)A very simple method, which only consider the shortest path length between each logical bits. Swap through the connection matrix to each
shortest mapping.
2)Using Deep Q Learning.
3)Combine the Monto Carlo Tree Search, Reinforcement Learning and Residual Network.

How to run:
There are three entrances, main, DQN_train and train.
Main is used for showing the results after training.
DQN_train is used for the training of DQN network.
Train is used for training of alpha zero based AI.
You should install pytorch v0.1.12.
The main file haven't finished, so you can only see the demo result in console window.
The you can changed the depth or circuit size in those entrance files.

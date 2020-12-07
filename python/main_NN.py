import pickle
import data
import torch
import torch.nn
import learning
import voxels
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import random

from NN import *
from RNN import *


def main(args):

    project_folder = "veleval"

    cuda = False
    #  gpx_pathfindind_cycling
    with open(args.path+"files/"+project_folder+"/data_processed/osmnx_pathfinding_simplified.df",'rb') as infile:
        df_pathfinding = pickle.load(infile)
    with open(args.path+"files/"+project_folder+"/data_processed/observations_matched_simplified.df",'rb') as infile:
        df_simplified = pickle.load(infile)
    with open(args.path+"files/"+project_folder+"/clustering/dbscan_observations.tab",'rb') as infile:
        tab_clusters = pickle.load(infile)
    with open(args.path+"files/"+project_folder+"/clustering/voxels_clustered_osmnx.dict",'rb') as infile:
        dict_voxels = pickle.load(infile)

    df = df_pathfinding

    tab_routes_voxels, _, _ = voxels.generate_voxels(df, df.iloc[0]["route_num"], df.iloc[-1]["route_num"])

    tab_routes_voxels_int = []
    
    df_voxels = pd.DataFrame()

    df_voxels_train = pd.DataFrame()
    df_voxels_test = pd.DataFrame()

    max_cluster = max(tab_clusters)+1

    for i in range(len(tab_routes_voxels)):
        nb_vox = 0
        tab_routes_voxels_int.append([])
        route = tab_routes_voxels[i]
        for vox in route:
            if(nb_vox%args.voxels_frequency==0): #(len(tab_routes_voxels_int[i])==0 or tab_routes_voxels_int[i][-1][0] != dict_voxels[vox]["cluster"]): 
                points = [dict_voxels[vox]["cluster"]]
                tab_routes_voxels_int[i].append(points)
            nb_vox += 1

        df_temp = pd.DataFrame(tab_routes_voxels_int[i], dtype=object)
        df_temp["route_num"] = i
        df_voxels = df_voxels.append(df_temp)
        
        proba_test = random.random()
        if(proba_test >= args.percentage_test/100):
            df_voxels_train = df_voxels_train.append(df_temp)
        else:
            df_voxels_test = df_voxels_test.append(df_temp)

    #print(len(df_voxels), len(df_voxels_train), len(df_voxels_test))

    df_train = df_voxels_train
    df_test = df_voxels_test
    
    if(len(df_test) == 0):
        df_test = df_train

    size_data = 1

    learning_rate = args.lr


    fc = NN(size_data, max_cluster)
    rnn = RNN(size_data, max_cluster)
    lstm = RNN_LSTM(size_data, max_cluster, args.hidden_size, args.num_layers, args.bidirectional)


    network = lstm

    if(cuda):
        network = network.cuda()

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    loss = nn.NLLLoss()

    tab_loss, tab_predict = learning.train(df_train, tab_clusters, loss, optimizer, network, size_data, cuda, args.num_samples, df_test)


    g_predict = learning.test(df_test, None, tab_clusters, size_data, cuda)
    print("Random:", g_predict*100, "%")

    g_predict = learning.test(df_test, network, tab_clusters, size_data, cuda)
    print("Good predict:", g_predict*100, "%")
    
    if(g_predict > 0.8):
        print("Saving network...")
        data.check_file("files/"+project_folder+"/neural_networks/network_temp.pt", [])
        torch.save(network.state_dict(), args.path+"files/"+project_folder+"/neural_networks/network_temp.pt")
        
    
    plt.plot(tab_loss)
    plt.ylabel('Error')
    plt.show()
    
    plt.plot(tab_predict[0], color='blue', label='train')
    plt.plot(tab_predict[1], color='red', label='test')
    plt.legend(loc='upper right')
    plt.ylabel('Prediction')
    plt.show()

    '''import torch
    import torch.nn as nn

    lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
    inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

    # initialize the hidden state.
    hidden = (torch.randn(1, 1, 3),
            torch.randn(1, 1, 3))
    for i in inputs:
        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
        out, hidden = lstm(i.view(1, 1, -1), hidden)
    print(out)

    # alternatively, we can do the entire sequence all at once.
    # the first value returned by LSTM is all of the hidden states throughout
    # the sequence. the second is just the most recent hidden state
    # (compare the last slice of "out" with "hidden" below, they are the same)
    # The reason for this is that:
    # "out" will give you access to all hidden states in the sequence
    # "hidden" will allow you to continue the sequence and backpropagate,
    # by passing it as an argument  to the lstm at a later time
    # Add the extra 2nd dimension
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
    out, hidden = lstm(inputs, hidden)
    print(out[4])'''

if __name__ == "__main__": 
    parse = argparse.ArgumentParser()
    parse.add_argument('--path', type=str, default="./", help="path to the project's main folder")
    parse.add_argument('--voxels-frequency', type=int, default=4, help="frequency of voxels to send to the network")
    parse.add_argument('--num-layers', type=int, default=2, help="number of layers in the LSTM network")
    parse.add_argument('--hidden-size', type=int, default=256, help="size of the hidden layer(s) in the network")
    parse.add_argument('--num-samples', type=int, default=75000, help="number of data (chosen randomly) to send to the network")
    parse.add_argument('--lr', type=float, default=5e-4, help='learning rate of the algorithm')
    parse.add_argument('--percentage-test', type=int, default=0, help='percentage of data to use as testing')
    parse.add_argument('--bidirectional', type=bool, default=False, help='change the LSTM in a bidirectional one')
    
    main(parse.parse_args())

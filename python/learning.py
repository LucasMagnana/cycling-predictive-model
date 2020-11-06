
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import data as data
import numpy as np
import random
from NN import *
from RNN import *
import datetime

def train_full_connected(df, tab_clusters, loss, optimizer, network, size_routes, cuda, nb_step):
    loss_tab = []
    pca = PCA(n_components=1)
    
    for _ in range(nb_step//30):
        input = []
        target = []
        for _ in range(30):
            num_route = random.randint(0, len(tab_clusters)-1)
            key = tab_clusters[num_route]
            if(key != -1):
                route = data.dataframe_to_array(df[df["route_num"]==num_route+1])
                data.normalize_route(route, size_routes)
                route = pca.fit_transform(route)
                input.append(route)
                target.append(key)
        input = torch.Tensor(input).squeeze(2)
        if(cuda):
            input = input.cuda()
        optimizer.zero_grad()
        output = network(input)
        target = torch.Tensor(target).long()
        if(cuda):
            output = output.cuda()
            target = target.cuda()
        l = loss(output, target)
        loss_tab.append(l.item())
        l.backward()
        optimizer.step()
    return(loss_tab)


def test_full_connected(df, network, dict_cluster, size_routes, cuda):
    good_predict = 0
    nb_predict = 0
    pca = PCA(n_components=1)
    input = []
    for key in dict_cluster:
        if(key != -1):
            tab_routes = []
            for num_route in dict_cluster[key]:
                route = data.dataframe_to_array(df[df["route_num"]==num_route+1])
                data.normalize_route(route, size_routes)
                route = pca.fit_transform(route)
                tab_routes.append(route)
            input = torch.Tensor(tab_routes).squeeze(2)
            if(cuda):
                input = input.cuda()
            output = network(input)
            pred = output.argmax(dim=1, keepdim=True)
            for i in range(pred.shape[0]):
                if(key == pred[i].item()):
                    good_predict += 1
                nb_predict += 1

    return good_predict/nb_predict


def train_recursive(df, tab_clusters, loss, optimizer, network, size_data, cuda, nb_step, df_test=None):
    loss_tab = []
    predict_tab = []
    print("start:", datetime.datetime.now().time())
    for s in range(nb_step):

        if(s != 0 and s%(nb_step//4) == 0):
            print("1/4:", datetime.datetime.now().time())

        if(s>nb_step):
            predict_tab.append(test_recursive(df_test, network, tab_clusters, size_data, cuda))

        key = -1
        route = []
        while(key == -1 or len(route) == 0):
            num_route = random.randint(df.iloc[0]["route_num"], df.iloc[-1]["route_num"])
            key = tab_clusters[num_route]
            route = data.dataframe_to_array(df[df["route_num"]==num_route], size_data)
            
        tens_route = torch.Tensor(route).unsqueeze(1)
        if(cuda):
            tens_route = tens_route.cuda()

        hidden = network.initHidden()
        target = torch.Tensor([key]).long()
        #network.zero_grad()
        optimizer.zero_grad()
        for i in range(tens_route.shape[0]):
            if(isinstance(network, RNN_LSTM)):
                input = tens_route[i].unsqueeze(1)
            else:
                input = tens_route[i]
            output, hidden = network(input, hidden)
        if(cuda):
            output = output.cuda()
            target = target.cuda()
        l = loss(output, target)
        loss_tab.append(l.item())
        l.backward()
        optimizer.step()
        '''for p in network.parameters():
            p.data.add_(-0.005, p.grad.data)'''
    return loss_tab, predict_tab

def test_recursive(df, network, tab_clusters, size_data, cuda):
    good_predict = 0
    nb_predict = 0
    for i in range(df.iloc[0]["route_num"], df.iloc[-1]["route_num"]+1):
        route = data.dataframe_to_array(df[df["route_num"]==i], size_data)
        if(tab_clusters[i] != -1 and len(route)>0):
            tens_route = torch.Tensor(route).unsqueeze(1)
            if(cuda):
                tens_route = tens_route.cuda()

            hidden = network.initHidden()
            for j in range(tens_route.shape[0]):
                if(isinstance(network, RNN_LSTM)):
                    input = tens_route[j].unsqueeze(1)
                else:
                    input = tens_route[j]
                output, hidden = network(input, hidden)
            pred = output.argmax(dim=1, keepdim=True)
            if(tab_clusters[i] == pred.item()):
                good_predict += 1
            nb_predict += 1
    return good_predict/nb_predict


def test_random(df, tab_clusters):
    good_predict = 0
    nb_predict = 0
    last_clust = max(tab_clusters)
    for i in range(df.iloc[-1]["route_num"]+1):
        if(tab_clusters[i] != -1):
            pred = random.randint(0, last_clust)
            if(tab_clusters[i] == pred):
                good_predict += 1
            nb_predict += 1
    return good_predict/nb_predict


def train(df, tab_clusters, loss, optimizer, network, size_data, cuda, nb_step, df_test=None):
    if(isinstance(network, RNN) or isinstance(network, RNN_LSTM)):
        return train_recursive(df, tab_clusters, loss, optimizer, network, size_data, cuda, nb_step, df_test)
    else:
        return train_full_connected(df, tab_clusters, loss, optimizer, network, size_data, cuda, nb_step)

def test(df, network, tab_clusters, size_data, cuda):
    if(isinstance(network, RNN) or isinstance(network, RNN_LSTM)):
        return test_recursive(df, network, tab_clusters, size_data, cuda)
    elif(isinstance(network, NN)):
        return test_full_connected(df, network, tab_clusters, size_data, cuda)
    else:
        return test_random(df, tab_clusters)

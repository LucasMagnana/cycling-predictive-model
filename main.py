import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
import json
import torch
import torch.nn as nn
from math import sin, cos, sqrt, atan2, radians
import copy
from sklearn.cluster import *
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import random
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

import datetime

import python.data as data
import python.display as dp
import python.voxels as voxel
import python.metric as metric
import python.clustering as cl
import python.RNN as RNN
import python.validation as validation
#import python.learning as learning
#from python.NN import *


pd.options.mode.chained_assignment = None

project_folder = "veleval"


with open("files/"+project_folder+"/data_processed/observations_matched_simplified.df",'rb') as infile:
    df_simplified = pickle.load(infile)
with open("files/"+project_folder+"/data_processed/osmnx_pathfinding_simplified.df",'rb') as infile:
    df_pathfinding = pickle.load(infile)       
tab_routes_voxels_pathfinding, _, _ = voxel.generate_voxels(df_pathfinding, df_pathfinding.iloc[0]["route_num"], df_pathfinding.iloc[-1]["route_num"])

with open("files/"+project_folder+"/city_graphs/city.ox", "rb") as infile:
    G_1 = pickle.load(infile)
with open("files/"+project_folder+"/city_graphs/city.ox", "rb") as infile:
    G_base_1 = pickle.load(infile)
nodes_1, _ = ox.graph_to_gdfs(G_1)
tree_1 = KDTree(nodes_1[['y', 'x']], metric='euclidean')

if(project_folder == "veleval"):
    with open("files/"+project_folder+"/city_graphs/city.ox", "rb") as infile:
        G_2 = pickle.load(infile)
    with open("files/"+project_folder+"/city_graphs/city.ox", "rb") as infile:
        G_base_2 = pickle.load(infile)
    nodes_2, _ = ox.graph_to_gdfs(G_2)
    tree_2 = KDTree(nodes_2[['y', 'x']], metric='euclidean')




data.check_file("files/"+project_folder+"/city_graphs/graph_modifications.dict", {})
with open("files/"+project_folder+"/city_graphs/graph_modifications.dict",'rb') as infile:
    dict_modif = pickle.load(infile)


with open("./files/"+project_folder+"/data_processed/osmnx_pathfinding_simplified.df",'rb') as infile:
    df_pathfinding = pickle.load(infile)
with open("./files/"+project_folder+"/clustering/dbscan_observations.tab",'rb') as infile:
    tab_clusters = pickle.load(infile)
with open("./files/"+project_folder+"/clustering/dbscan_observations.dict",'rb') as infile:
    dict_cluster = pickle.load(infile)
dict_clusters = cl.tab_clusters_to_dict(tab_clusters)
with open("./files/"+project_folder+"/clustering/voxels_clustered_osmnx.dict",'rb') as infile:
    dict_voxels_clustered = pickle.load(infile)
with open("./files/"+project_folder+"/neural_networks/saved/network.param",'rb') as infile:
    param = pickle.load(infile)
with open("./files/"+project_folder+"/neural_networks/saved/num_test.tab",'rb') as infile:
    tab_num_test = pickle.load(infile)
with open("files/"+project_folder+"/clustering/kmeans_voxels_osmnx.sk",'rb') as infile:
    kmeans = pickle.load(infile)


size_data = 1

network = RNN.RNN_LSTM(size_data, max(tab_clusters)+1, param.hidden_size, param.num_layers, param.bidirectional, param.dropout)
network.load_state_dict(torch.load("files/"+project_folder+"/neural_networks/saved/network_temp.pt"))
network.eval()


deviation = 0 #5e-3

tab_coeff_simplified = []
tab_coeff_modified = []

tab_diff_coeff = []
#________________________________________________________________________


for i in tab_num_test: #len(tab_clusters)):
    print(i)
    df_temp = df_pathfinding[df_pathfinding["route_num"]==i]
    d_point = [df_temp.iloc[0]["lat"], df_temp.iloc[0]["lon"]]
    f_point = [df_temp.iloc[-1]["lat"], df_temp.iloc[-1]["lon"]]
    rand = random.uniform(-deviation, deviation)
    d_point[0] += rand
    rand = random.uniform(-deviation, deviation)
    d_point[1] += rand
    rand = random.uniform(-deviation, deviation)
    f_point[0] += rand
    rand = random.uniform(-deviation, deviation)
    f_point[1] += rand

    if(project_folder == "veleval" and d_point[0] <= 45.5):
        G = G_2
        nodes = nodes_2
        tree = tree_2
        G_base = G_base_2
    else:
        G = G_1
        nodes = nodes_1
        tree = tree_1
        G_base = G_base_1

    df_route, cl, nb_new_cluster = validation.find_cluster(d_point, f_point, network, param.voxels_frequency, df_pathfinding, dict_voxels_clustered, 
                                kmeans, tree, G, nodes)
    #print(cl, tab_clusters[i])
    if(cl == tab_clusters[i]):
        print("good predict")
        #dp.display_cluster_heatmap_mapbox(df_simplified, dict_cluster[cl])
    #dp.display_mapbox(df_route)
    #dp.display(df_temp)


    ################################################################################_
    df_route = df_route[["lat", "lon", "route_num"]]

    df_cluster = pd.DataFrame(columns=["lat", "lon", "route_num"])
    for num_route in range(len(dict_cluster[cl])):
        df_temp = df_simplified[df_simplified["route_num"]==dict_cluster[cl][num_route]]
        df_temp["route_num"] = num_route
        df_cluster = df_cluster.append(df_temp)
    _, _, dict_voxels_cluster = voxel.generate_voxels(df_cluster, df_cluster.iloc[0]["route_num"], df_cluster.iloc[-1]["route_num"])

    df_temp = df_simplified[df_simplified["route_num"]==i]
    df_temp["route_num"] = 1
    df_temp["type"] = 0
    df_route["type"] = 2
    df_c_simplified = df_route.append(df_temp)

    tab_voxels, tab_voxels_global, dict_voxels = voxel.generate_voxels(df_c_simplified, 0, 1)

    tab_voxels_min_route = voxel.get_voxels_with_min_routes(dict_voxels, 2)
    df = pd.DataFrame(tab_voxels_min_route, columns=["lat", "lon", "route_num", "type"])
    df_c_simplified = df_c_simplified.append(df)

    #dp.display_mapbox(df_c_simplified, color="type") 

    coeff_simplified = metric.get_distance_voxels(0, 1, tab_voxels)
    
    if(cl in dict_modif):
        for key in dict_modif[cl]:
            vertexes = key.split(";")
            v = int(vertexes[0])
            v_n = int(vertexes[1])
            G[v][v_n][0]['length'] -= G[v][v_n][0]['length']*(dict_modif[cl][key]/1.6)
    else :
        print("start:", datetime.datetime.now().time())
        dict_modif[cl] = {}
        for v in G:
            for v_n in G[v]:
                df_line = pd.DataFrame([[G.nodes[v]['y'], G.nodes[v]['x'], 0], [G.nodes[v_n]['y'], G.nodes[v_n]['x'], 0]], columns=["lat", "lon", "route_num"])
                tab_voxels, _, _ = voxel.generate_voxels(df_line, 0, 0)
                nb_vox_found = 0
                tot_coeff = 0
                for vox in tab_voxels[0]:
                    if vox in dict_voxels_cluster:
                        nb_vox_found += 1
                        tot_coeff += dict_voxels_cluster[vox]["cyclability_coeff"]
                if(nb_vox_found > 0):
                    tot_coeff /= nb_vox_found
                    dict_modif[cl][str(v)+";"+str(v_n)] = tot_coeff
                    #print(dict_modif[cl][str(v)+";"+str(v_n)], G[v][v_n][0]['length'], G[v][v_n][0]['length']*(tot_coeff/1.6))
                    G[v][v_n][0]['length'] -= G[v][v_n][0]['length']*(tot_coeff/1.6)
        print("end:", datetime.datetime.now().time())
        with open("files/"+project_folder+"/city_graphs/graph_modifications.dict",'wb') as outfile:
            pickle.dump(dict_modif, outfile)



    route = data.pathfind_route_osmnx(d_point, f_point, tree, G, nodes)
    route_coord = [[G.nodes[x]["x"], G.nodes[x]["y"]] for x in route]
    route_coord = [x + [1, 2] for x in route_coord]
    df_route_modified = pd.DataFrame(route_coord, columns=["lon", "lat", "route_num", "type"])

    #print(df_route_modified)
    #dp.display_mapbox(df_route_modified)

    for key in dict_modif[cl]:
        vertexes = key.split(";")
        v = int(vertexes[0])
        v_n = int(vertexes[1])
        G[v][v_n][0]['length'] = G_base[v][v_n][0]['length']

    df_c_modified = df_simplified[df_simplified["route_num"]==i]
    df_c_modified["route_num"] = 0
    df_c_modified["type"] = 0
    df_c_modified = df_c_modified.append(df_route_modified)


    tab_voxels, tab_voxels_global, dict_voxels = voxel.generate_voxels(df_c_modified, 0, 1)
    tab_voxels_min_route = voxel.get_voxels_with_min_routes(dict_voxels, 2)
    df = pd.DataFrame(tab_voxels_min_route, columns=["lat", "lon", "route_num", "type"])
    df_c_modified = df_c_modified.append(df)

    #dp.display_mapbox(df_c_modified, color="type") 


    coeff_modified = metric.get_distance_voxels(0, 1, tab_voxels_global)
    

    tab_coeff_simplified.append(1-min(coeff_simplified))
    tab_coeff_modified.append(1-min(coeff_modified))

    tab_diff_coeff.append((1-min(coeff_modified))-(1-min(coeff_simplified)))

    print(tab_diff_coeff[-1])

    #print(1-min(coeff_simplified) <= 1-min(coeff_modified))

    #print(1-min(coeff_simplified), 1-min(coeff_modified))

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(25,15))
ax = plt.axes()
x = np.linspace(0, len(tab_diff_coeff), len(tab_diff_coeff))
plt.plot(x, tab_diff_coeff, color='red', linewidth=3.5)
plt.show()
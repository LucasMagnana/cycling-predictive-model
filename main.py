import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
import json
import torch
import torch.nn as nn
from math import sin, cos, sqrt, atan2, radians, exp
import copy
from sklearn.cluster import *
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import random
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.neighbors import KDTree
import os
import argparse

import datetime

import python.data as data
import python.display as dp
import python.voxels as voxel
import python.metric as metric
import python.clustering as cl
import python.RNN as RNN
import python.validation as validation
import python.graphs as graphs

if __name__ == "__main__": 
    parse = argparse.ArgumentParser()
    parse.add_argument('--global-metric', type=bool, default=True, help="whether to use the global metric or not")
    parse.add_argument('--project-folder', type=str, default="veleval", help='folder of the project')
    
args = parse.parse_args()


project_folder = args.project_folder

global_metric = args.global_metric


def create_dict_modif(G, dict_cluster, df_simplified):
    dict_modif = {}
    print("start:", datetime.datetime.now().time())
    dict_dict_voxels_cluster = {}
    for cl in dict_cluster:
        if(cl > -1):
            df_cluster = pd.DataFrame(columns=["lat", "lon", "route_num"])
            for num_route in range(len(dict_cluster[cl])):
                df_temp = df_simplified[df_simplified["route_num"]==dict_cluster[cl][num_route]]
                df_temp["route_num"] = num_route
                df_cluster = df_cluster.append(df_temp)
            _, dict_voxels_cluster_global, dict_voxels_cluster = voxel.generate_voxels(df_cluster, df_cluster.iloc[0]["route_num"], df_cluster.iloc[-1]["route_num"])
            dict_dict_voxels_cluster[cl] = dict_voxels_cluster
    for v in G:
        for v_n in G[v]:
            df_line = pd.DataFrame([[G.nodes[v]['y'], G.nodes[v]['x'], 0], [G.nodes[v_n]['y'], G.nodes[v_n]['x'], 0]], columns=["lat", "lon", "route_num"])
            tab_voxels, _, _ = voxel.generate_voxels(df_line, 0, 0)
            for cl in dict_dict_voxels_cluster:
                nb_vox_found = 0
                tot_coeff = 0
                dict_voxels_cluster = dict_dict_voxels_cluster[cl]
                for vox in tab_voxels[0]:
                    if vox in dict_voxels_cluster:
                        nb_vox_found += 1
                        tot_coeff += dict_voxels_cluster[vox]["cyclability_coeff"]
                if(nb_vox_found > 0):
                    if(cl not in dict_modif):
                        dict_modif[cl] = {}
                    tot_coeff /= nb_vox_found
                    dict_modif[cl][str(v)+";"+str(v_n)] = tot_coeff
    print("end:", datetime.datetime.now().time())
    return dict_modif


pd.options.mode.chained_assignment = None

with open("files/"+project_folder+"/data_processed/mapbox_pathfinding.df",'rb') as infile:
    df_mapbox_routes_test = pickle.load(infile)

with open("files/"+project_folder+"/data_processed/observations_matched_simplified.df",'rb') as infile:
    df_simplified = pickle.load(infile)


if not os.path.exists(os.path.dirname("files/"+project_folder+"/city_graphs/city.ox")):
    print("Creating city.ox")
    exec(open("files/"+project_folder+"/load_city.py").read())

with open("files/"+project_folder+"/city_graphs/city.ox", "rb") as infile:
    G_1 = pickle.load(infile)
with open("files/"+project_folder+"/city_graphs/city.ox", "rb") as infile:
    G_base_1 = pickle.load(infile)
nodes_1, _ = ox.graph_to_gdfs(G_1)
tree_1 = KDTree(nodes_1[['y', 'x']], metric='euclidean')


if("veleval" in project_folder):
    with open("files/"+project_folder+"/city_graphs/city_2.ox", "rb") as infile:
        G_2 = pickle.load(infile)
    with open("files/"+project_folder+"/city_graphs/city_2.ox", "rb") as infile:
        G_base_2 = pickle.load(infile)
nodes_2, _ = ox.graph_to_gdfs(G_2)
tree_2 = KDTree(nodes_2[['y', 'x']], metric='euclidean')


#print(len(G_1), len(G_2))

with open("./files/"+project_folder+"/clustering/voxels_clustered_osmnx.dict",'rb') as infile:
    dict_voxels_clustered = pickle.load(infile)
with open("files/"+project_folder+"/clustering/kmeans_voxels_osmnx.sk",'rb') as infile:
    kmeans = pickle.load(infile)
with open("./files/"+project_folder+"/neural_networks/saved/num_test.tab",'rb') as infile:
    tab_num_test = pickle.load(infile)
with open("./files/"+project_folder+"/clustering/dbscan_observations.tab",'rb') as infile:
    tab_clusters = pickle.load(infile)
dict_cluster = cl.tab_clusters_to_dict(tab_clusters)
for key in dict_cluster:
    for nr in dict_cluster[key]:
        if nr in tab_num_test:
            dict_cluster[key].remove(nr)
 

data.check_file("files/"+project_folder+"/data_processed/unreachable_routes.tab", [[],[]])
with open("files/"+project_folder+"/data_processed/unreachable_routes.tab",'rb') as infile:
    tab_unreachable_routes = pickle.load(infile) 
    
    

if(not(os.path.isfile("files/"+project_folder+"/city_graphs/graph_modifications.dict"))):
    dict_modif = create_dict_modif(G_1, dict_cluster, df_simplified)
    dict_modif_se = create_dict_modif(G_2, dict_cluster, df_simplified)
    print(len(dict_modif), len(dict_modif_se))
    for cl in dict_modif_se:
        if(cl not in dict_modif):
            dict_modif[cl] = dict_modif_se[cl]
            
    print(len(dict_modif))
        
    with open("files/"+project_folder+"/city_graphs/graph_modifications.dict",'wb') as outfile:
        pickle.dump(dict_modif, outfile)
   

            
if(not(os.path.isfile("files/"+project_folder+"/city_graphs/graph_modifications_global.dict"))):    
    dict_modif_global = {}
        
    dict_cluster_global = {0: range(len(tab_clusters)+1)}
    dict_modif_global[0] = create_dict_modif(G_1, dict_cluster_global, df_simplified)[0]
    print(len(dict_modif_global[0]))
    dict_modif_global[1] = create_dict_modif(G_2, dict_cluster_global, df_simplified)[0]
    print(len(dict_modif_global[1]))
    with open("files/"+project_folder+"/city_graphs/graph_modifications_global.dict",'wb') as outfile:
        pickle.dump(dict_modif_global, outfile)
    
            
with open("files/"+project_folder+"/city_graphs/graph_modifications.dict",'rb') as infile:
    dict_modif = pickle.load(infile)

with open("files/"+project_folder+"/city_graphs/graph_modifications_global.dict",'rb') as infile:
    dict_modif_global = pickle.load(infile)
    

def create_path_compute_similarity(d_point, f_point, df, tree, G, nodes, global_metric):

    route = data.pathfind_route_osmnx(d_point, f_point, tree, G, nodes)
    route_coord = [[G.nodes[x]["y"], G.nodes[x]["x"]] for x in route]
    route_coord = [x + [0] for x in route_coord]

    df_route = pd.DataFrame(route_coord, columns=["lat", "lon", "route_num"])
    df_route = data.rd_compression(df_route, 0, 1)
   

    tab_route_voxels, _, _ = voxel.generate_voxels(df_route, 0, 0)
    df["route_num"] = 1
    df["type"] = 0     
    df_route["type"] = 2
    df_coeff = df_route.append(df)

    tab_voxels, tab_voxels_global, dict_voxels = voxel.generate_voxels(df_coeff, 0, 1)

    #\\\DEBUG///
    '''tab_voxels_min_route = voxel.get_voxels_with_min_routes(dict_voxels, 2, global_metric)
    df = pd.DataFrame(tab_voxels_min_route, columns=["lat", "lon", "route_num", "type"])
    df_coeff = df_coeff.append(df)
    dp.display_mapbox(df_coeff, color="type")'''

    if(global_metric):
        coeff = metric.get_distance_voxels(0, 1, tab_voxels_global)
    else:
        coeff = metric.get_distance_voxels(0, 1, tab_voxels)

    return df_route, tab_route_voxels[0], coeff


def modify_network_graph(cl, dict_modif, G, coeff_diminution = 1):
    for key in dict_modif[cl]:
        vertexes = key.split(";")
        v = int(vertexes[0])
        v_n = int(vertexes[1]) 
        if(v in G):
            G[v][v_n][0]['length'] -= G[v][v_n][0]['length']*min(1, exp(dict_modif[cl][key])-1)
        else: 
            return False
    return True



def cancel_network_graph_modifications(cl, dict_modif, G, G_base):
    for key in dict_modif[cl]:
        vertexes = key.split(";")
        v = int(vertexes[0])
        v_n = int(vertexes[1])
        if(v in G):
            G[v][v_n][0]['length'] = G_base[v][v_n][0]['length']
        else:
            break
        
def choose_route_endpoints(df_route, num_route, deviation):
        global tab_unreachable_routes
        d_point = [df_route.iloc[0]["lat"], df_route.iloc[0]["lon"]]
        if(num_route in tab_unreachable_routes[0]):
            d_point = [df_route.iloc[1]["lat"], df_route.iloc[1]["lon"]]
        f_point = [df_route.iloc[-1]["lat"], df_route.iloc[-1]["lon"]]
        if(num_route in tab_unreachable_routes[1]):
            f_point = [df_route.iloc[-2]["lat"], df_route.iloc[-2]["lon"]]
        rand = random.uniform(-deviation, deviation)
        d_point[0] += rand
        rand = random.uniform(-deviation, deviation)
        d_point[1] += rand
        rand = random.uniform(-deviation, deviation)
        f_point[0] += rand
        rand = random.uniform(-deviation, deviation)
        f_point[1] += rand
        
        return d_point, f_point


def main_global(global_metric):

    deviation = 0 #5e-2

    tab_coeff_simplified = []
    tab_coeff_modified = []

    tab_diff_coeff = []



    modify_network_graph(0, dict_modif_global, G_1)
    modify_network_graph(1, dict_modif_global, G_2)
    
    for i in tab_num_test:
        df_route_tested = df_simplified[df_simplified["route_num"]==i]

        d_point, f_point = choose_route_endpoints(df_route_tested, i, deviation)

        if("veleval" in project_folder and df_route_tested.iloc[0]["lat"] <= 45.5):
            G = G_2
            nodes = nodes_2
            tree = tree_2
            G_base = G_base_2
        else:
            G = G_1
            nodes = nodes_1
            tree = tree_1
            G_base = G_base_1

        df_route, tab_route_voxels, coeff_simplified = create_path_compute_similarity(d_point, f_point, df_route_tested, tree, G_base, nodes, global_metric)
        df_route, tab_route_voxels, coeff_modified = create_path_compute_similarity(d_point, f_point, df_route_tested, tree, G, nodes, global_metric)

        tab_coeff_simplified.append(1-min(coeff_simplified))
        tab_coeff_modified.append(1-min(coeff_modified))
        tab_diff_coeff.append((1-min(coeff_modified))-(1-min(coeff_simplified)))
    
    
    print("GLOBAL :")
    print("Mean shortest path similarity:", sum(tab_coeff_simplified)/len(tab_coeff_simplified)*100, "%")
    print("Mean modified path similarity:", sum(tab_coeff_modified)/len(tab_coeff_modified)*100, "%")
    print("Mean improvement:", sum(tab_diff_coeff)/len(tab_diff_coeff)*100, "%")
    print("===============================")

    return tab_coeff_simplified, tab_coeff_modified, tab_diff_coeff
        

def main_clusters(global_metric, deviation=0):


    tab_coeff_simplified = []
    tab_coeff_modified = []

    tab_diff_coeff = []

    for key in dict_cluster:
        if(key != -1):
            df_temp = df_simplified[df_simplified["route_num"]==dict_cluster[key][0]]
            if("veleval" in project_folder and df_temp.iloc[0]["lat"] <= 45.5):
                G = G_2
                nodes = nodes_2
                tree = tree_2
                G_base = G_base_2
            else:
                G = G_1
                nodes = nodes_1
                tree = tree_1
                G_base = G_base_1

            modify_network_graph(key, dict_modif, G)
    
    for i in tab_num_test: #len(tab_clusters)):e
        df_route_tested = df_simplified[df_simplified["route_num"]==i]
        
        
        d_point, f_point = choose_route_endpoints(df_route_tested, i, deviation)

        if("veleval" in project_folder and df_route_tested.iloc[0]["lat"] <= 45.5):
            G = G_2
            nodes = nodes_2
            tree = tree_2
            G_base = G_base_2
        else:
            G = G_1
            nodes = nodes_1
            tree = tree_1
            G_base = G_base_1

        df_route, tab_route_voxels, coeff_simplified = create_path_compute_similarity(d_point, f_point, df_route_tested, tree, G_base, nodes, global_metric)
        df_route, tab_route_voxels, coeff_modified = create_path_compute_similarity(d_point, f_point, df_route_tested, tree, G, nodes, global_metric)

        tab_coeff_simplified.append(1-min(coeff_simplified))
        tab_coeff_modified.append(1-min(coeff_modified))
        tab_diff_coeff.append((1-min(coeff_modified))-(1-min(coeff_simplified)))


    print("CLUSTERS ONLY:")
    print("Mean shortest path similarity:", sum(tab_coeff_simplified)/len(tab_coeff_simplified)*100, "%")
    print("Mean modified path similarity:", sum(tab_coeff_modified)/len(tab_coeff_modified)*100, "%")
    print("Mean improvement:", sum(tab_diff_coeff)/len(tab_diff_coeff)*100, "%")
    print("===============================")
    

    return tab_coeff_simplified, tab_coeff_modified, tab_diff_coeff

    




def main_clusters_NN(global_metric, deviation = 0, full_print=False):
    


    with open("./files/"+project_folder+"/neural_networks/saved/network.param",'rb') as infile:
        param = pickle.load(infile)

    size_data = 1

    network = RNN.RNN_LSTM(size_data, max(tab_clusters)+1, param.hidden_size, param.num_layers, param.bidirectional, param.dropout)
    network.load_state_dict(torch.load("files/"+project_folder+"/neural_networks/saved/network_temp.pt"))
    network.eval()

    nb_good_predict = 0

    tab_coeff_simplified = [[], []]
    tab_coeff_modified = [[], []]

    tab_diff_coeff = [[], []]

    for i in tab_num_test: #len(tab_clusters)):
        good_predict = False
        df_route_tested = df_simplified[df_simplified["route_num"]==i]
        
        d_point, f_point = choose_route_endpoints(df_route_tested, i, deviation)

        if("veleval" in project_folder and df_route_tested.iloc[0]["lat"] <= 45.5):
            G = G_2
            nodes = nodes_2
            tree = tree_2
            G_base = G_base_2
        else:
            G = G_1
            nodes = nodes_1
            tree = tree_1
            G_base = G_base_1

        df_route, tab_route_voxels, coeff_simplified = create_path_compute_similarity(d_point, f_point, df_route_tested, tree, G, nodes, global_metric)

        cl, nb_new_cluster = validation.find_cluster(tab_route_voxels, network, param.voxels_frequency, dict_voxels_clustered, 
                                    kmeans, df_route_tested)
        #print(cl, tab_clusters[i])
        if(cl == tab_clusters[i]):
            #print("good predict")
            nb_good_predict += 1
            good_predict = True
            #dp.display_cluster_heatmap_mapbox(df_simplified, dict_cluster[cl])
        #dp.display_mapbox(df_route)
        #dp.display(df_route_tested)


        ################################################################################_
        df_route = df_route[["lat", "lon", "route_num"]]

        modify_network_graph(cl, dict_modif, G)


        df_route_modified,_,coeff_modified = create_path_compute_similarity(d_point, f_point, df_simplified[df_simplified["route_num"]==i], tree, G, nodes, global_metric)

        cancel_network_graph_modifications(cl, dict_modif, G, G_base)
        

        if(good_predict):
            tab_coeff_simplified[0].append(1-min(coeff_simplified))
            tab_coeff_modified[0].append(1-min(coeff_modified))
            tab_diff_coeff[0].append((1-min(coeff_modified))-(1-min(coeff_simplified)))
        else:
            tab_coeff_simplified[1].append(1-min(coeff_simplified))
            tab_coeff_modified[1].append(1-min(coeff_modified))
            tab_diff_coeff[1].append((1-min(coeff_modified))-(1-min(coeff_simplified)))

        #print(tab_diff_coeff[-1])

        #print(1-min(coeff_simplified) <= 1-min(coeff_modified))

        #print(1-min(coeff_simplified), 1-min(coeff_modified))
        
    if(full_print):
        print("===============================")
        print("GOOD PREDICTIONS :")
        print("===============================")
        print("Mean shortest path similarity:", sum(tab_coeff_simplified[0])/len(tab_coeff_simplified[0])*100, "%")
        print("Mean modified path similarity:", sum(tab_coeff_modified[0])/len(tab_coeff_modified[0])*100, "%")
        print("Mean improvement:", sum(tab_diff_coeff[0])/len(tab_diff_coeff[0])*100, "%")
        print("===============================")
        print("BAD PREDICTIONS :")
        print("===============================")
        print("Mean shortest path similarity:", sum(tab_coeff_simplified[1])/len(tab_coeff_simplified[1])*100, "%")
        print("Mean modified path similarity:", sum(tab_coeff_modified[1])/len(tab_coeff_modified[1])*100, "%")
        print("Mean improvement:", sum(tab_diff_coeff[1])/len(tab_diff_coeff[1])*100, "%")
        print("===============================")
        
    print("NN + CLUSTERS (Good predict:", len(tab_coeff_simplified[0])/(len(tab_coeff_simplified[0])+len(tab_coeff_simplified[1]))*100, "%):")
    print("===============================")
    print("Mean shortest path similarity:", sum(sum(tab_coeff_simplified,[]))/sum(len(row) for row in tab_coeff_simplified)*100, "%")
    print("Mean modified path similarity:", sum(sum(tab_coeff_modified,[]))/sum(len(row) for row in tab_coeff_modified)*100, "%")
    print("Mean improvement:", sum(sum(tab_diff_coeff,[]))/sum(len(row) for row in tab_diff_coeff)*100, "%")


    return tab_coeff_simplified, tab_coeff_modified, tab_diff_coeff




def main_clusters_full_predict(global_metric, deviation = 0):

    tab_coeff_simplified = []
    tab_coeff_modified = []

    tab_diff_coeff = []

    for i in tab_num_test: #len(tab_clusters)):
        df_route_tested = df_simplified[df_simplified["route_num"]==i]
        df_mapbox = df_mapbox_routes_test[df_mapbox_routes_test["route_num"]==i]
        
        d_point, f_point = choose_route_endpoints(df_route_tested, i, deviation)

        if("veleval" in project_folder and df_route_tested.iloc[0]["lat"] <= 45.5):
            G = G_2
            nodes = nodes_2
            tree = tree_2
            G_base = G_base_2
        else:
            G = G_1
            nodes = nodes_1
            tree = tree_1
            G_base = G_base_1

        df_route, tab_route_voxels, coeff_simplified = create_path_compute_similarity(d_point, f_point, df_route_tested, tree, G, nodes, global_metric)


        ################################################################################_
        df_route = df_route[["lat", "lon", "route_num"]]

        modify_network_graph(tab_clusters[i], dict_modif, G)


        df_route_modified,_,coeff_modified = create_path_compute_similarity(d_point, f_point, df_simplified[df_simplified["route_num"]==i], tree, G, nodes, global_metric)

        cancel_network_graph_modifications(tab_clusters[i], dict_modif, G, G_base)
        
        tab_coeff_simplified.append(1-min(coeff_simplified))
        tab_coeff_modified.append(1-min(coeff_modified))
        tab_diff_coeff.append((1-min(coeff_modified))-(1-min(coeff_simplified)))
        
        '''df_route_tested["route_num"] = 0
        df_route_modified["route_num"] = 1
        df_mapbox["route_num"] = 2
        df_route["route_num"] = 3
        
        df_display = df_route_tested.append(df_route_modified)
        df_display = df_display.append(df_route)
        df_display = df_display.append(df_mapbox)
        dp.display_mapbox(df_display, color="route_num")'''

       
    
    print("CLUSTERS FULL PREDICTED:")
    print("Mean shortest path similarity:", sum(tab_coeff_simplified)/len(tab_coeff_simplified)*100, "%")
    print("Mean modified path similarity:", sum(tab_coeff_modified)/len(tab_coeff_modified)*100, "%")
    print("Mean improvement:", sum(tab_diff_coeff)/len(tab_diff_coeff)*100, "%")
    print("===============================")

    return tab_coeff_simplified, tab_coeff_modified, tab_diff_coeff







def main_mapbox(global_metric):
    global df_simplified
    global df_mapbox_routes_test
        
    tab_coeff_modified = []
    
    
    for i in tab_num_test:
        df_observation = df_simplified[df_simplified["route_num"]==i]
        df_mapbox = df_mapbox_routes_test[df_mapbox_routes_test["route_num"]==i]

        df_observation["route_num"] = 0
        df_observation["type"] = 0    
        df_mapbox["route_num"] = 1
        df_mapbox["type"] = 2
        
        df_coeff = df_observation.append(df_mapbox)

        tab_voxels, tab_voxels_global, dict_voxels = voxel.generate_voxels(df_coeff, 0, 1)

        #\\\DEBUG///
        '''tab_voxels_min_route = voxel.get_voxels_with_min_routes(dict_voxels, 2)
        df = pd.DataFrame(tab_voxels_min_route, columns=["lat", "lon", "route_num", "type"])
        df_coeff = df_coeff.append(df)
        dp.display_mapbox(df_coeff, color="type") '''

        if(global_metric):
            coeff = metric.get_distance_voxels(0, 1, tab_voxels_global)
        else:
            coeff = metric.get_distance_voxels(0, 1, tab_voxels)
        
        tab_coeff_modified.append(1-min(coeff))


        '''df_observation["route_num"] = 0
        df_mapbox["route_num"] = 1
        
        df_display = df_observation.append(df_mapbox)
        dp.display_mapbox(df_display, color="route_num")'''
        
            
    print("MAPBOX :")
    print("Mean modified path similarity:", sum(tab_coeff_modified)/len(tab_coeff_modified)*100, "%")
    print("===============================")
        
    return tab_coeff_modified



tab_results_base = []
tab_results_improvement = []


tab_coeff_simplified, tab_coeff_modified, tab_diff_coeff = main_clusters_full_predict(global_metric)


tab_coeff_modified = main_mapbox(global_metric)
tab_results_base.append(sum(tab_coeff_modified)/len(tab_coeff_modified)*100)
tab_results_improvement.append(0)

tab_coeff_simplified, tab_coeff_modified, tab_diff_coeff = main_clusters(global_metric)
tab_results_base.append(sum(tab_coeff_simplified)/len(tab_coeff_simplified)*100)
tab_results_improvement.append(sum(tab_diff_coeff)/len(tab_diff_coeff)*100)

G_1 = deepcopy(G_base_1)
G_2 = deepcopy(G_base_2)


tab_coeff_simplified, tab_coeff_modified, tab_diff_coeff = main_global(global_metric)

G_1 = deepcopy(G_base_1)
G_2 = deepcopy(G_base_2)

tab_results_base.append(sum(tab_coeff_simplified)/len(tab_coeff_simplified)*100)
tab_results_improvement.append(sum(tab_diff_coeff)/len(tab_diff_coeff)*100)


tab_coeff_simplified, tab_coeff_modified, tab_diff_coeff = main_clusters_NN(global_metric)



tab_results_base.append(sum(sum(tab_coeff_simplified,[]))/sum(len(row) for row in tab_coeff_simplified)*100)
tab_results_improvement.append(sum(sum(tab_diff_coeff,[]))/sum(len(row) for row in tab_diff_coeff)*100)


tab_results_NN = [[sum(tab_coeff_simplified[0])/len(tab_coeff_simplified[0])*100, sum(tab_coeff_simplified[1])/len(tab_coeff_simplified[1])*100,
sum(sum(tab_coeff_simplified,[]))/sum(len(row) for row in tab_coeff_simplified)*100], [sum(tab_diff_coeff[0])/len(tab_diff_coeff[0])*100, 
sum(tab_diff_coeff[1])/len(tab_diff_coeff[1])*100, sum(sum(tab_diff_coeff,[]))/sum(len(row) for row in tab_diff_coeff)*100]]

  
    
graphs.similarity_results_grpah(tab_results_base, tab_results_improvement, project_folder, global_metric)

graphs.NN_results_graph(tab_results_NN, project_folder, global_metric)

    

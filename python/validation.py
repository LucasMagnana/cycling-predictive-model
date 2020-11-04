import pickle
import pandas as pd
import argparse
import torch
import random

import python.voxels as voxels
import python.data as data


def find_cluster(d_point, f_point, network, voxels_frequency, df, dict_voxels, clustering, tree, G, cuda):

    nb_new_cluster = 0

    route = data.pathfind_route_osmnx(d_point, f_point, tree, G)
    route_coord = [[G.nodes[x]["x"], G.nodes[x]["y"]] for x in route]
    route_coord = [x + [0] for x in route_coord]

    df_route = pd.DataFrame(route_coord, columns=["lon", "lat", "route_num"])
    tab_routes_voxels, _, _ = voxels.generate_voxels(df_route, 0, 0)
    route = tab_routes_voxels[0]

    tab_voxels_int = []
    nb_vox = 0

    for vox in route:
        if(nb_vox%voxels_frequency==0):
            vox_str = vox.split(";")
            vox_int = [int(vox_str[0]), int(vox_str[1])]
            tab_points = voxels.get_voxel_points(vox_int)
            if vox in dict_voxels:
                cl = dict_voxels[vox]["cluster"]
            else:
                cl = clustering.predict([[tab_points[0][0], tab_points[0][1], 0]])[0]
                nb_new_cluster += 1
            points = [cl]
            tab_voxels_int.append(points)
        nb_vox += 1

    route = tab_voxels_int
    if(len(route)>0):
        tens_route = torch.Tensor(route).unsqueeze(1)
        if(cuda):
            tens_route = tens_route.cuda()

        hidden = network.initHidden()
        for j in range(tens_route.shape[0]):
            input = tens_route[j].unsqueeze(1)
            output, hidden = network(input, hidden)
        pred = output.argmax(dim=1, keepdim=True)
    
    return df_route, pred.item(), nb_new_cluster
    

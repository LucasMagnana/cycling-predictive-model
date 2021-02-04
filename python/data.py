import json 
import pandas as pd
import pickle
import sys
from rdp import *
import requests
import numpy as np
import xml.etree.ElementTree as ET
import os
from math import sin, cos, sqrt, atan2, radians
from geopy.distance import geodesic
import networkx as nx
import osmnx as ox
from sklearn.neighbors import KDTree
from datetime import datetime
#import python.voxels as voxel

token = "pk.eyJ1IjoibG1hZ25hbmEiLCJhIjoiY2s2N3hmNzgwMGNnODNqcGJ1N2l2ZXZpdiJ9.-aOxDLM8KbEQnJfXegtl7A"

def check_file(file, content):
    if(not(os.path.isfile(file))):
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        print("Warning: creating", file)
        open(file, "x")
        with open(file,'wb') as infile:
            pickle.dump(content, infile)
        

def request_map_matching(df_route):
    route = df_route.to_numpy()
    coord=""
    tab_requests = []
    i=0
    for i in range(len(route)):
        coord += str(route[i][1])+","+str(route[i][0])+";"
        if(i!=0 and i%99 == 0):
            coord = coord[:-1]
            tab_requests.append(requests.get("https://api.mapbox.com/matching/v5/mapbox/cycling/"+coord+"?access_token="+token))
            coord = ""
    if(i!=0 and i%99 != 0):
        coord = coord[:-1]
        tab_requests.append(requests.get("https://api.mapbox.com/matching/v5/mapbox/cycling/"+coord+"?access_token="+token))
    return tab_requests


def clean_dataframe(df, tab_unreachable_routes=None):
    print("Cleaning dataframe...")
    nb_empty = 0
    df_final = pd.DataFrame(columns=df.columns)
    for i in range(df.iloc[0]["route_num"], df.iloc[-1]["route_num"]+1):
        df_temp = df[df["route_num"]==i]
        if(len(df_temp)<=1):
            nb_empty += 1
        else:
            df_temp["route_num"] = i-nb_empty
            df_final = df_final.append(df_temp)
            if(tab_unreachable_routes != None):
                if(i in tab_unreachable_routes[0]):
                    tab_unreachable_routes[0].remove(i)
                    tab_unreachable_routes[0].append(i-nb_empty)
                if(i in tab_unreachable_routes[1]):
                    tab_unreachable_routes[1].remove(i)
                    tab_unreachable_routes[1].append(i-nb_empty)

    return df_final


def load_bikepath_lyon(file):
    with open(file) as infile:
        data = json.load(infile)        
    df_bikepath = pd.DataFrame(columns=['lat','lon', 'route_num'])
    for i in range(len(data["features"])):
        route = data["features"][i]["geometry"]["coordinates"]
        while(len(route[0]) != 2 or not(isinstance(route[0][0], float))):
            route = route[0]
        df_temp = pd.DataFrame(route, columns=['lon','lat'])
        df_temp
        df_temp['route_num'] = i
        df_bikepath = df_bikepath.append(df_temp)
    return df_bikepath


def simplify_gps(infile, outfile, nb_routes=sys.maxsize, dim=2):
    if(nb_routes > 0):
        with open(infile,'rb') as infile:
            df = pickle.load(infile)
        check_file(outfile, pd.DataFrame(columns=df.columns))
        with open(outfile,'rb') as infile:
            df_simplified = pickle.load(infile)
        if(len(df_simplified) == 0):
            last_route_simplified = 0
        else:
            last_route_simplified = df_simplified.iloc[-1]["route_num"]+1
        nb_routes = min(df.iloc[-1]["route_num"] - last_route_simplified, nb_routes)
        print(last_route_simplified, last_route_simplified+nb_routes+1)
        df_simplified = df_simplified.append(rd_compression(df, last_route_simplified, last_route_simplified+nb_routes+1, dim))
        with open(outfile, 'wb') as outfile:
            pickle.dump(df_simplified, outfile)



def rd_compression(df, start, end, dim=2, eps=1e-4):
    """
    Compress a dataframe with douglas-peucker's algorithm.
    Parameters
    ----------
    df : pandas' DataFrame with columns=['lat', 'lon', 'route_num']
        Dataframe to compress
    eps : int in [0, 1[ , optional
        Precision of the compression (high value = few points)
    nb_routes : int
        Number of routes to compress
    Returns
    -------
    pandas' DataFrame with columns=['lat', 'lon', 'route_num']
        the compressed DataFrame
    """
    df_simplified = pd.DataFrame()
    for i in range(start, end):
        route = df[df['route_num']==i].values
        if(len(route)>0):
            simplified = np.delete(route, range(dim, route.shape[1]), 1)
            simplified = rdp(simplified.tolist(), epsilon=eps)
            if(dim == 2):
                df_temp = pd.DataFrame(simplified, columns=['lat', 'lon'])
            else:
                df_temp = pd.DataFrame(simplified, columns=['lat', 'lon', 'time_elapsed'])
            df_temp["route_num"]=route[0][-1]
            if(len(df_temp) == 0):
                print(i, "bite")
            df_simplified = df_simplified.append(df_temp)
    return df_simplified


def mapmatching(infile_str, outfile_str, nb_routes=sys.maxsize):
    if(nb_routes > 0):
        with open(infile_str,'rb') as infile:
            df = pickle.load(infile)
        check_file(outfile_str, pd.DataFrame(columns=['lat', 'lon', 'route_num']))
        with open(outfile_str,'rb') as infile:
            df_map_matched = pickle.load(infile)

        if(df_map_matched.empty):
            begin = 0
        else:
            begin = df_map_matched.iloc[-1]["route_num"]+1
        for i in range(begin, min(begin+1+nb_routes, df.iloc[-1]["route_num"]+1)):
            distance = 0
            df_temp = df[df["route_num"]==i]
            tab_requests = request_map_matching(df_temp)
            tab_points = []
            for req in tab_requests:
                response = req.json()
                if("tracepoints" in response):
                    route = response["tracepoints"]
                    for point in route:
                        if(point != None):
                            tab_points.append([point['location'][1], point['location'][0], i])
                            distance += point['distance']
            df_map_matched = df_map_matched.append(pd.DataFrame(tab_points, columns=["lat", "lon", "route_num"]))
            with open(outfile_str, 'wb') as outfile:
                pickle.dump(df_map_matched, outfile)


def request_route(lat1, long1, lat2, long2, mode="cycling"):
    coord = str(long1)+","+str(lat1)+";"+str(long2)+","+str(lat2)
    return requests.get("https://api.mapbox.com/directions/v5/mapbox/"+mode+"/"+coord, 
                            params={"alternatives": "true", "geometries": "geojson", "steps": "true", "access_token": token}) 


def pathfinding_mapbox(infilestr, outfilestr, tabnumteststr=None, nb_routes=sys.maxsize):
    if(nb_routes > 0):
        with open(infilestr,'rb') as infile:
            df_map_matched_simplified = pickle.load(infile)
        check_file(outfilestr, pd.DataFrame(columns=['lat', 'lon', 'route_num']))
        with open(outfilestr,'rb') as infile:
            df_pathfinding = pickle.load(infile)
        if(tabnumteststr != None):
            with open(tabnumteststr,'rb') as infile:
                tab_num_test = pickle.load(infile)
        else:
            tab_num_test = None

        if(len(df_pathfinding) == 0):
            last_route_pathfound = 0
        else:
            last_route_pathfound = df_pathfinding.iloc[-1]["route_num"]+1

        nb_routes = min(df_map_matched_simplified.iloc[-1]["route_num"] - last_route_pathfound, nb_routes)
        print(last_route_pathfound, last_route_pathfound+nb_routes+1)
        for i in range(last_route_pathfound, last_route_pathfound+nb_routes+1):
            if(tab_num_test == None or i in tab_num_test):
                print(i)
                df_temp = df_map_matched_simplified[df_map_matched_simplified["route_num"]==i]
                d_point = [df_temp.iloc[0]["lat"], df_temp.iloc[0]["lon"]]
                f_point = [df_temp.iloc[-1]["lat"], df_temp.iloc[-1]["lon"]]
                df_temp = pathfind_route_mapbox(d_point, f_point, df_pathfinding, i)
                df_pathfinding = df_pathfinding.append(df_temp)
                with open(outfilestr, 'wb') as outfile:
                    pickle.dump(df_pathfinding, outfile)


def pathfind_route_mapbox(d_point, f_point, df_pathfinding=pd.DataFrame(), num_route=1):
    save_route = True
    req = request_route(d_point[0], d_point[1], f_point[0], f_point[1]) #mapbox request to find a route between the stations
    response = req.json()
    if(response['code']=='Ok'): #if a route have been found
        steps = response['routes'][0]['legs'][0]['steps'] #we browse all the steps of the route
        for step in steps:
            if(step['maneuver']['instruction'].find("Wharf") != -1):
                save_route = False #if the route is not good (using a boat) we don't save it
                break
        if(save_route): #if we save the route
            df_temp = pd.DataFrame.from_records(response['routes'][0]['geometry']['coordinates'], 
                                    columns=['lon', 'lat']) #create a DF from the route (nparray)
            df_temp["route_num"] = num_route
            return df_temp[["lat", "lon", "route_num"]]
    return None




def pathfinding_osmnx(infile_str, outfile_str, graphfile_str, unreachableroutesfile_str, nb_routes=sys.maxsize):
    if(nb_routes > 0):
        with open(infile_str,'rb') as infile:
            df_simplified = pickle.load(infile)

        with open(graphfile_str+"/city.ox",'rb') as infile:
            G = pickle.load(infile)
            nodes, _ = ox.graph_to_gdfs(G)
            tree = KDTree(nodes[['y', 'x']], metric='euclidean')

        check_file(outfile_str, pd.DataFrame(columns=['lat', 'lon', 'route_num']))
        with open(outfile_str,'rb') as infile:
            df_pathfinding = pickle.load(infile)
            
        check_file(unreachableroutesfile_str, [[],[]])
        with open(unreachableroutesfile_str,'rb') as infile:
            tab_unreachable_routes = pickle.load(infile)

        if(len(df_pathfinding) == 0):
            last_route_pathfound = 0
        else:
            last_route_pathfound = df_pathfinding.iloc[-1]["route_num"]+1

        nb_routes = min(df_simplified.iloc[-1]["route_num"] - last_route_pathfound, nb_routes)
        print(last_route_pathfound, last_route_pathfound+nb_routes+1)
        for i in range(last_route_pathfound, last_route_pathfound+nb_routes+1):
            print(i)
            df_temp = df_simplified[df_simplified["route_num"]==i]
            d_point = [df_temp.iloc[0]["lat"], df_temp.iloc[0]["lon"]]
            if(i in tab_unreachable_routes[0]):
                d_point = [df_temp.iloc[1]["lat"], df_temp.iloc[1]["lon"]]
            f_point = [df_temp.iloc[-1]["lat"], df_temp.iloc[-1]["lon"]]
            if(i in tab_unreachable_routes[1]):
                f_point = [df_temp.iloc[-2]["lat"], df_temp.iloc[-2]["lon"]]
            route = pathfind_route_osmnx(d_point, f_point, tree, G, nodes)
            route_coord = [[G.nodes[x]["y"], G.nodes[x]["x"]] for x in route]
            route_coord = [x + [i] for x in route_coord]
            df_pathfinding = df_pathfinding.append(pd.DataFrame(route_coord, columns=["lat", "lon", "route_num"]))
            with open(outfile_str, 'wb') as outfile:
                pickle.dump(df_pathfinding, outfile)
                
                



def pathfinding_osmnx_veleval(infile_str, outfile_str, graphfile_str, unreachableroutesfile_str, nb_routes=sys.maxsize):
    if(nb_routes > 0):
        with open(infile_str,'rb') as infile:
            df_simplified = pickle.load(infile)

        with open(graphfile_str+"/city.ox",'rb') as infile:
            G = pickle.load(infile)
            nodes, _ = ox.graph_to_gdfs(G)
            tree = KDTree(nodes[['y', 'x']], metric='euclidean')

        with open(graphfile_str+"/city_2.ox",'rb') as infile:
            G_1 = pickle.load(infile)
            nodes_1, _ = ox.graph_to_gdfs(G_1)
            tree_1 = KDTree(nodes_1[['y', 'x']], metric='euclidean')

        check_file(outfile_str, pd.DataFrame(columns=['lat', 'lon', 'route_num']))
        with open(outfile_str,'rb') as infile:
            df_pathfinding = pickle.load(infile)
            
        check_file(unreachableroutesfile_str, [[],[]])
        with open(unreachableroutesfile_str,'rb') as infile:
            tab_unreachable_routes = pickle.load(infile)

        if(len(df_pathfinding) == 0):
            last_route_pathfound = 0
        else:
            last_route_pathfound = df_pathfinding.iloc[-1]["route_num"]+1

        nb_routes = min(df_simplified.iloc[-1]["route_num"] - last_route_pathfound, nb_routes)
        print(last_route_pathfound, last_route_pathfound+nb_routes+1)
        for i in range(last_route_pathfound, last_route_pathfound+nb_routes+1):
            print(i)
            df_temp = df_simplified[df_simplified["route_num"]==i]
            d_point = [df_temp.iloc[0]["lat"], df_temp.iloc[0]["lon"]]
            if(i in tab_unreachable_routes[0]):
                print("chatte")
                d_point = [df_temp.iloc[1]["lat"], df_temp.iloc[1]["lon"]]
            f_point = [df_temp.iloc[-1]["lat"], df_temp.iloc[-1]["lon"]]
            if(i in tab_unreachable_routes[1]):
                print("bite")
                f_point = [df_temp.iloc[-2]["lat"], df_temp.iloc[-2]["lon"]]
            if(d_point[0] > 45.5):
                route = pathfind_route_osmnx(d_point, f_point, tree, G, nodes)
                route_coord = [[G.nodes[x]["y"], G.nodes[x]["x"]] for x in route]
                route_coord = [x + [i] for x in route_coord]
            else:
                route = pathfind_route_osmnx(d_point, f_point, tree_1, G_1, nodes_1)
                route_coord = [[G_1.nodes[x]["y"], G_1.nodes[x]["x"]] for x in route]
                route_coord = [x + [i] for x in route_coord]
            df_pathfinding = df_pathfinding.append(pd.DataFrame(route_coord, columns=["lat", "lon", "route_num"]))
            with open(outfile_str, 'wb') as outfile:
                pickle.dump(df_pathfinding, outfile)



def pathfind_route_osmnx(d_point, f_point, tree, G, nodes):
    d_idx = tree.query([d_point], k=1, return_distance=False)[0]
    f_idx = tree.query([f_point], k=1, return_distance=False)[0]
    closest_node_to_d = nodes.iloc[d_idx].index.values[0]
    closest_node_to_f = nodes.iloc[f_idx].index.values[0]
    route = nx.shortest_path(G, 
                             closest_node_to_d,
                             closest_node_to_f,
                             weight='length')
    return route





def distance_between_points(p1, p2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(p1[0])
    lon1 = radians(p1[1])
    lat2 = radians(p2[0])
    lon2 = radians(p2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance


def compute_distance(infile, outfile):
    with open(infile,'rb') as infile:
        df = pickle.load(infile)
    tab_distances = []
    for i in range(df.iloc[-1]["route_num"]+1):
        df_temp = df[df["route_num"]==i]
        dist = 0
        if(df_temp.shape[0] >= 2):
            for j in range(df_temp.shape[0]-1):
                dist += distance_between_points(df_temp.iloc[j], df_temp.iloc[j+1])
                #dist += geodesic((df_temp.iloc[j][0],df_temp.iloc[j][1]), (df_temp.iloc[j+1][0], df_temp.iloc[j+1][1])).kilometers
            tab_distances.append(dist)
        else:
            tab_distances.append(0)
    with open(outfile, 'wb') as outfile:
        pickle.dump(tab_distances, outfile)



def dataframe_to_array(df, n=2):
    tab = df.to_numpy().tolist()
    for i in range(len(tab)):
        tab[i] = tab[i][:n]
    return tab


def harmonize_route(v1, v2):
    diff = max(len(v1), len(v2)) - min(len(v1), len(v2))
    if(len(v1) > len(v2)):
        shorter_route = v2
    else:
        shorter_route = v1
    for i in range(0, diff*2, 2):
        indice = i%(len(shorter_route)-1)
        new_point = [(shorter_route[indice][0]+shorter_route[indice+1][0])/2,
                     (shorter_route[indice][1]+shorter_route[indice+1][1])/2]
        shorter_route.insert(indice+1, new_point)


def normalize_route(v1, n):
    diff = n-len(v1)
    if(diff < 0):
        print("Route is too long !")
        return
    for i in range(0, diff*2, 2):
        indice = i%(len(v1)-1)
        new_point = [(v1[indice][0]+v1[indice+1][0])/2,
                     (v1[indice][1]+v1[indice+1][1])/2]
        v1.insert(indice+1, new_point)


def add_time_elapsed(file, nb_routes=1):
    if(nb_routes > 0):
        with open(file,'rb') as infile:
            df = pickle.load(infile)
        if("time_elapsed" not in df.columns):
            print("Warning : Adding time elapsed...")
            tab_te = compute_time_elapsed(df)
            if(len(tab_te)==len(df)):
                df.insert(loc=2, column='time_elapsed', value=tab_te)
                with open(file,'wb') as outfile:
                    pickle.dump(df, outfile)
            else:
                print("Dimension error.")

    
def compute_time_elapsed(df):
    tab_te = []
    for i in range(df.iloc[-1]["route_num"]+1):
        df_temp = df[df["route_num"]==i]
        tab_te.append(0)
        last_line = pd.DataFrame()
        for line in df_temp.iloc:
            if(last_line.empty):
                last_line = line
            else:
                time = pd.Timedelta(line["time"] - last_line["time"]).total_seconds()
                tab_te.append(tab_te[-1]+time)
                last_line = line
    return tab_te


def add_speed(file):
    with open(file,'rb') as infile:
        df = pickle.load(infile)
    if("speed" not in df.columns):
        print("Warning : Adding speed...")
        tab_speed = compute_speed(df)
        if(len(tab_speed)==len(df)):
            df.insert(loc=3, column='speed', value=tab_speed)
            with open(file,'wb') as outfile:
                pickle.dump(df, outfile)
            '''df = df[df["speed"]>= 0]
            df = clean_dataframe(df)
            with open(file,'wb') as outfile:
                pickle.dump(df, outfile)'''
        else:
            print("Dimension error.")


def compute_speed(df):
    tab_speed = []
    for i in range(df.iloc[0]["route_num"], df.iloc[-1]["route_num"]+1):
        df_temp = df[df["route_num"]==i]
        tab_speed.append(0)
        last_line = pd.DataFrame()
        for line in df_temp.iloc:
            if(last_line.empty):
                last_line = line
            else:
                distance = geodesic((last_line[0],last_line[1]), (line[0], line[1])).meters
                time = line["time_elapsed"]-last_line["time_elapsed"]

                if(time == 0):
                    print("bite", i)
                    tab_speed.append(-1)
                else:
                    tab_speed.append(distance/time)


                last_line = line
    return tab_speed


def load_veleval():
    df = pd.DataFrame()
    for i in range(1, 1110):
        tree = ET.parse('data/veleval/GPX/data'+str(i)+'.gpx')
        if(len(tree.getroot()) > 1):
            route = []
            root = tree.getroot()[1][0]
            df_temp = pd.DataFrame(columns=['lat', 'lon'])
            for child in root:
                j=0
                while("time" not in child[j].tag):
                    j+=1
                time = datetime.strptime(child[j].text, '%Y-%m-%dT%H:%M:%S.%fZ')
                coord = child.attrib
                coord['lat'] = float(coord['lat'])
                coord['lon'] = float(coord['lon'])
                route.append([coord['lat'], coord['lon'], time, i-1])
            df = df.append(pd.DataFrame(route, columns=["lat", "lon", "time", "route_num"]))
    return df


def compute_average_speed(df, nb_points):
    tab_speed = df["speed"].values

    tab_last_speed = []
    i=0

    tab_avg_speed = []
    for speed in tab_speed:
        avg_speed = 0
        if (len(tab_last_speed) == nb_points):
            i+=1
            if(i>=len(tab_last_speed)):
                i=0
            tab_last_speed[i] = speed
        else:
            i+=1
            tab_last_speed.append(speed)
            
        for last_speed in tab_last_speed:
            avg_speed += last_speed
            
        avg_speed /= len(tab_last_speed)
        tab_avg_speed.append(avg_speed)
    
    return tab_avg_speed

def compute_average_acceleration(df, nb_points):
    tab_speed= df["speed"].values

    tab_last_accel = []
    i=0
    last_speed = 0
    tab_avg_accel = []
    for speed in tab_speed:
        avg_accel = 0
        if (len(tab_last_accel) == nb_points):
            i+=1
            if(i>=len(tab_last_accel)):
                i=0
            tab_last_accel[i] = speed-last_speed
        else:
            i+=1
            tab_last_accel.append(speed-last_speed)
            
        for last_accel in tab_last_accel:
            avg_accel += last_accel
            
        avg_accel /= len(tab_last_accel)
        tab_avg_accel.append(avg_accel)

        last_speed = speed
    
    return tab_avg_accel


'''def bikepath_fusion(infile, outfile, nb_routes=1):
    if(nb_routes > 0):
        with open(infile,'rb') as infile:
                df_bikepath = pickle.load(infile)
        check_file(outfile, pd.DataFrame(columns=['lon', 'lat', 'route_num']))
        with open(outfile,'rb') as infile:
            df_bikepath_fusioned = pickle.load(infile)
        route_num_fusioned = 0
        nb_changes = 0
        nb_routes = df_bikepath.iloc[-1]["route_num"]
        for i in range(nb_routes+1):
            if(len(df_bikepath[df_bikepath["route_num"]==i]) > 0):
                print(i)
                for j in range(i, nb_routes):
                    if(len(df_bikepath[df_bikepath["route_num"]==j]) > 0):
                        p1 = df_bikepath[df_bikepath["route_num"]==i].values.tolist()[-1][:2]
                        p2 = df_bikepath[df_bikepath["route_num"]==j].values.tolist()[0][:2]
                        v1 = voxel.find_voxel_int(p1)
                        v2 = voxel.find_voxel_int(p2)
                        if(v1 == v2):
                            nb_changes += 1
                            df_bikepath = df_bikepath.replace({"route_num": j}, i)                              
            df_temp = df_bikepath[df_bikepath["route_num"]==i]
            df_temp["route_num"] = route_num_fusioned
            route_num_fusioned += 1
            df_bikepath_fusioned = df_bikepath_fusioned.append(df_temp)    
        print(nb_changes, "changes")
        with open(outfile,'wb') as outfile:
            pickle.dump(df_bikepath_fusioned, outfile)


def bikepath_fusion_first(df_bikepath):
    verbose = False
    n_route = 0
    n_route_next = n_route+1
    n = n_route
    nb_change = 0
    while(n_route < df_bikepath.iloc[-1]["route_num"]):
        p1 = df_bikepath[df_bikepath["route_num"]==n_route].values.tolist()[-1][:2]
        p2 = df_bikepath[df_bikepath["route_num"]==n_route_next].values.tolist()[0][:2]
        v1 = voxel.find_voxel_int(p1)
        v2 = voxel.find_voxel_int(p2)
        if(v1 == v2):
            tab_changes = [n_route]
            while(v1 == v2):
                tab_changes.append(n_route_next)
                n_route_next += 1
                n_route += 1
                p1 = df_bikepath[df_bikepath["route_num"]==n_route].values.tolist()[-1][:2]
                p2 = df_bikepath[df_bikepath["route_num"]==n_route_next].values.tolist()[0][:2]
                v1 = voxel.find_voxel_int(p1)
                v2 = voxel.find_voxel_int(p2)
            for i in range(len(tab_changes)):
                df_bikepath = df_bikepath.replace({"route_num": tab_changes[i]}, n)
                if(verbose):
                    print(tab_changes[i], "->", n, "equals")
                if(i != 0):
                    nb_change += 1
            n+=1
            n_route += 1
        else:
            df_bikepath = df_bikepath.replace({"route_num": n_route}, n)
            if(verbose):
                print(n_route, "->", n)
            n_route += 1
            n += 1
        n_route_next += 1
    df_bikepath = df_bikepath.replace({"route_num": df_bikepath.iloc[-1]["route_num"]}, n)
    print(nb_change, "changes")
    return df_bikepath
    
#df_bikepath_fusioned = bikepath_fusion(df_bikepath)
        



def harmonize_route(v1, v2):
    diff = max(len(v1), len(v2)) - min(len(v1), len(v2))
    if(len(v1)>len(v2)):
        v2 += [v2[-1]]*diff
    else:
        v1 += [v1[-1]]*diff
    return v1, v2'''
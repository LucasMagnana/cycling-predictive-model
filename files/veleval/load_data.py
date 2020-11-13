import gpxpy
import gpxpy.gpx
import pandas as pd
import os
import pickle

num_route = 0
num_file = 0
df = pd.DataFrame(columns=["lat", "lon", "time", "route_num"])
for root, dirs, files in os.walk("./data/veleval/", topdown=False): #load all the data
    for name in files: 
            if(".gpx" in name):
                with open(root+'/'+name) as gpx_file:
                    track_added = False
                    num_file += 1
                    print(num_file, num_route)
                    gpx = gpxpy.parse(gpx_file)
                    route=[]
                    for track in gpx.tracks:
                        for segment in track.segments:
                            for point in segment.points:
                                route.append([point.latitude, point.longitude, point.time, num_route])
                    if(len(route)>2):
                        if(num_route==1):
                            print(route)
                        df = df.append(pd.DataFrame(route, columns=["lat", "lon", "time", "route_num"]))
                        num_route += 1
                        route=[]



                    for r in gpx.routes:
                        for point in r.points:
                            route.append([point.latitude, point.longitude, point.time, num_route])
                    if(len(route)>2):
                        if(num_route==1):
                            print(route)
                        df = df.append(pd.DataFrame(route, columns=["lat", "lon", "time", "route_num"]))
                        num_route += 1
                        route=[]

                    for waypoint in gpx.waypoints:
                        waypoint_added = True
                        route.append([waypoint.latitude, waypoint.longitude, waypoint.time, num_route])

                    if(len(route)>2):
                        if(num_route==1):
                            print(route)
                        df = df.append(pd.DataFrame(route, columns=["lat", "lon", "time", "route_num"]))
                        num_route += 1
                        route=[]
                        


with open("files/veleval/data_processed/observations_wtime.df", "wb") as infile:
    pickle.dump(df, infile)





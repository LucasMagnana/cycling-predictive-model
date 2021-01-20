import osmnx as ox
import pickle
import pandas as pd

with open("files/veleval_full/data_processed/observations_matched_simplified.df", "rb") as infile:
    df = pickle.load(infile)



df = df[["lat", "lon"]]
df_values = df.values

nort_lat_ly = -9999999
sout_lat_ly = 9999999
east_lon_ly = -9999999
west_lon_ly = 9999999

nort_lat_se = -9999999
sout_lat_se = 9999999
east_lon_se = -9999999
west_lon_se = 9999999

for coord in df_values:

    if(coord[0] > 45.5):
        if(coord[0]>nort_lat_ly):
            nort_lat_ly = coord[0]
        if(coord[0]<sout_lat_ly):
            sout_lat_ly = coord[0]

        if(coord[1]>east_lon_ly):
            east_lon_ly = coord[1]
        if(coord[1]<west_lon_ly):
            west_lon_ly = coord[1]
    else:
        if(coord[0]>nort_lat_se):
            nort_lat_se = coord[0]
        if(coord[0]<sout_lat_se):
            sout_lat_se = coord[0]

        if(coord[1]>east_lon_se):
            east_lon_se = coord[1]
        if(coord[1]<west_lon_se):
            west_lon_se = coord[1]

df_test = pd.DataFrame([[nort_lat_ly, west_lon_ly, 0],
[nort_lat_ly, east_lon_ly, 0],
[sout_lat_ly, east_lon_ly, 0],
[sout_lat_ly, west_lon_ly, 0],
[nort_lat_se, west_lon_se, 1],
[nort_lat_se, east_lon_se, 1],
[sout_lat_se, east_lon_se, 1],
[sout_lat_se, west_lon_se, 1]] ,columns=["lat", "lon", "route_num"])

#dp.display_mapbox(df_test)

    


G_lyon = ox.graph_from_bbox(nort_lat_ly, sout_lat_ly, east_lon_ly, west_lon_ly)
G_stetienne = ox.graph_from_bbox(nort_lat_se, sout_lat_se, east_lon_se, west_lon_se)

with open("files/veleval_full/city_graphs/city.ox", "wb") as outfile:
    pickle.dump(G_lyon, outfile)
with open("files/veleval_full/city_graphs/city_2.ox", "wb") as outfile:
    pickle.dump(G_stetienne, outfile)

import osmnx as ox

lyon = (45.74846, 4.84671)
st_etienne = (45.4333, 4.4)
G_lyon = ox.graph_from_point(lyon, distance=10000)
G_stetienne = ox.graph_from_point(st_etienne, distance=7500)
G = G_lyon

with open("files/veleval/city_graphs/city_1.ox", "wb") as outfile:
    pickle.dump(G_lyon, outfile)
with open("files/veleval/city_graphs/city_2.ox", "wb") as outfile:
    pickle.dump(G_stetienne, outfile)
import json
import pandas as pd
import pickle

tab_routes = []

with open('data/monresovelo/trip5000.json') as f:
  data = json.load(f)

for i in range(len(data["features"])):
  for coord in data["features"][i]["geometry"]["coordinates"]:
    tab_routes.append([coord[1], coord[0], i])

df = pd.DataFrame(tab_routes, columns=["lat", "lon", "route_num"])

print(df)

with open("files/monresovelo/data/observations.df", "wb") as infile:
  pickle.dump(df, infile)

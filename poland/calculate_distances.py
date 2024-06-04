import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from geopy.distance import geodesic
from itertools import combinations

df = pd.read_csv("cities.csv")
#100 largest cities
df = df.sort_values("population", ascending=False).head(100)

df = df[["latitude", "longitude"]]
data = df.values
num_rows = len(data)

print("Calculating distances...")
distance_matrix = np.zeros((num_rows, num_rows), dtype=np.float32)
for i in tqdm(range(num_rows)):
    for j in range(i, num_rows):
        distance_matrix[i, j] = distance_matrix[j, i] = geodesic(data[i], data[j]).meters

output_path = "distances.npy"
print(f"Saving distances to {output_path}")
np.save(output_path, distance_matrix)

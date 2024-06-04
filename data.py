import numpy as np
import pandas as pd
from pathlib import Path

def _into_distance_matrix(positions):
    n = len(positions)
    distance_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            distance_matrix[i, j] = distance_matrix[j, i] = np.sum((positions[i] - positions[j]) ** 2) ** 0.5
    return distance_matrix


def get_type1_data(n: int =100):
    populations = np.random.random(n)
    positions = np.random.random((n, 2))
    dm = _into_distance_matrix(positions)
    return populations, dm


def get_type2_data(n: int = 100):
    log_populations = np.random.normal(-0.85, 1.3, n)
    populations = np.exp(log_populations)
    positions = np.random.random((n, 2))
    dm = _into_distance_matrix(positions)
    return populations, dm


def get_poland_data():
    poland_cities_path = Path("./poland/cities.csv")
    poland_distances = Path("./poland/distances.npy")
    assert poland_cities_path.exists(), "Download the data first"
    assert poland_distances.exists(), "Calculate the distances first"

    df = pd.read_csv(poland_cities_path)
    populations = df["population"].values
    sorting_idx = populations.argsort()[::-1]
    top_100 = sorting_idx[:100]

    populations = populations[top_100]

    distance_matrix = np.load(poland_distances)

    mask = np.zeros(len(distance_matrix), dtype=bool)
    mask[top_100] = True

    distance_matrix = distance_matrix[mask].T[mask].T

    # Normalize the data
    populations = populations / populations.max()
    distnace_matrix = distance_matrix / distance_matrix.max()

    return populations, distnace_matrix


if __name__ == "__main__":
    print(get_poland_data())


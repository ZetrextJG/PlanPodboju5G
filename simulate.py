
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

from src import Solver, RandomWalk, WeightedRandomJumps, bit_random_state, uniform_random_states, heuristic_states
from data import get_type1_data, get_type2_data
from tqdm.auto import tqdm

lambdas = np.linspace(0.01, 1.5, 100)

n_cities = 100
n_iter = 100
n_steps = 1000

output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_id", type=int)
    args = parser.parse_args()
    assert args.lambda_id < len(lambdas)


    output_file = output_dir / f"lambda_{args.lambda_id}.json"

    lambdate = lambdas[args.lambda_id]
    records = []

    for _ in tqdm(range(n_iter)):
        populations, distance_matrix = get_type1_data()
        solver = Solver(populations, distance_matrix, RandomWalk)
        states = heuristic_states(n_cities, n_cities) # Special case because we start at different city each chain
        res = solver.simulate_chains(lambda_= lambdate, n_chains=100, steps=n_steps, states=states)

        records.append({
            "lambda": lambdate,
            "score": res["best_score"],
            "nb_states": res["best_state"].sum(),
            "population_sum": populations[res["best_state"]].sum(),
        })

    pd.DataFrame(records).to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    main()

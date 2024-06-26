from abc import ABC, abstractmethod
from functools import lru_cache
from numba import jit
from typing import Optional, Callable
import numpy as np


#######################
# Exploring chain
#######################


@lru_cache
@jit(nopython=True)
def get_geometric_probs(n: int, p: int) -> np.ndarray:
    vector = np.array([p]) ** np.arange(1, n + 1)
    return vector / np.sum(vector)


class MarkovChain(ABC):
    def __init__(self, n_cities: int) -> None:
        self.n_cities = n_cities
        assert self.n_cities > 0
        self.rng = np.random.default_rng()

    def validate_state(self, state: np.ndarray) -> bool:
        return state.sum() > 0

    @abstractmethod
    def next_state(self, state: np.ndarray) -> np.ndarray:
        ...


class RandomWalk(MarkovChain):
    def __init__(self, n_cities: int) -> None:
        super().__init__(n_cities)

    def next_state(self, state: np.ndarray) -> np.ndarray:
        u = self.rng.random()
        if u <= (1 / (self.n_cities + 1)):
            return state

        idx = self.rng.integers(0, self.n_cities, endpoint=False)
        new_state = state.copy()
        new_state[idx] = ~new_state[idx]
        if self.validate_state(new_state):
            return new_state
        return state


class WeightedRandomJumps(MarkovChain):
    def __init__(self, n_cities: int) -> None:
        super().__init__(n_cities)

    def next_state(self, state: np.ndarray) -> np.ndarray:
        n = self.n_cities
        probs = get_geometric_probs(n, 1 / 2)
        m = self.rng.choice(n, p=probs) + 1
        idxs = self.rng.choice(n, m, replace=False)

        new_state = state.copy()
        new_state[idxs] = ~new_state[idxs]
        if self.validate_state(new_state):
            return new_state
        return state


#######################
# Starting states
#######################


def bit_random_state(n_chains: int, n_cities: int) -> list[np.ndarray]:
    chains = []
    for _ in range(n_chains):
        chains.append(np.random.random(n_cities) < 0.5)
    return chains


def uniform_random_states(n_chains: int, n_cities: int) -> list[np.ndarray]:
    chains = []
    for _ in range(n_chains):
        m = np.random.binomial(n=n_cities, p=0.5)
        idxs = np.random.choice(n_cities, m, replace=False)
        mask = np.zeros(n_cities, dtype=bool)
        mask[idxs] = True
        chains.append(mask)
    return chains


def heuristic_states(n_chains: int, n_cities: int) -> list[np.ndarray]:
    """This returns exaclty n_chains states, each one with a different city selected."""
    assert n_chains == n_cities
    return [*np.eye(n_cities).astype(bool)]


#######################
# Metropolis-Hastings
#######################


class MetropolisHastings:
    states: list[np.ndarray]
    scores: np.ndarray

    def __init__(
        self,
        n_cities: int,
        n_chains: int,
        exploring_chain: MarkovChain,
        objective_function,
        states: Optional[list[np.ndarray]] = None,
    ):
        self.n_cities = n_cities
        self.n_chains = n_chains
        self.rng = np.random.default_rng()
        self.exploring_chain = exploring_chain
        self.objective_function = objective_function

        if states is not None:
            self.states = states
        else:
            self.states = bit_random_state(n_chains, n_cities)

        self.states = np.array(self.states, dtype=np.bool_)

        self.scores = [self.objective_function(state) for state in self.states]
        self.scores = np.array(self.scores)

    def acceptance(self, prev_score, new_score, beta: float = 1.0):
        return min(np.exp(beta * (new_score - prev_score)), 1)

    def forward(self, beta: float = 1.0):
        for i, state in enumerate(self.states):
            new_state = self.exploring_chain.next_state(state)
            new_score = self.objective_function(new_state)
            u = self.rng.random()
            if u <= self.acceptance(self.scores[i], new_score, beta=beta):
                self.states[i] = new_state
                self.scores[i] = new_score


#######################
# Solver
#######################


@jit(nopython=True)
def calculate_radius(distance_matrix: np.ndarray, state: np.ndarray) -> float:
    return distance_matrix[state].T[state].max() / 2


class Solver:
    def __init__(
        self,
        populations: np.ndarray,
        distance_matrix: np.ndarray,
        exploring_chain_type: type[MarkovChain],
    ):
        self.populations = populations
        self.distance_matrix = distance_matrix

        self.n_cities = len(populations)
        self.exploring_chain = exploring_chain_type(n_cities=self.n_cities)
        assert self.distance_matrix.shape == (self.n_cities, self.n_cities)

    def get_objective_funtion(self, lambda_: float) -> Callable[[np.ndarray], float]:
        def objective_function(state: np.ndarray) -> float:
            r = calculate_radius(self.distance_matrix, state)
            value = (
                np.sum(self.populations, where=state)
                - lambda_ * self.n_cities * np.pi * r**2
            )
            return value

        return objective_function

    def simulate_chains(
        self,
        lambda_: float = 0.1,
        n_chains: int = 10,
        steps: int = 1000,
        states: Optional[list[np.ndarray]] = None,
        beta: float = 1,
    ):
        mh = MetropolisHastings(
            n_cities=self.n_cities,
            n_chains=n_chains,
            exploring_chain=self.exploring_chain,
            objective_function=self.get_objective_funtion(lambda_),
            states=states,
        )

        best_state = None
        best_score = -np.inf

        step_best_scores = []
        step_best_state_size = []
        for _ in range(steps):
            mh.forward(beta=beta)
            idx = np.argmax(mh.scores)
            step_best_scores.append(mh.scores[idx])
            step_best_state_size.append(mh.states[idx].sum())
            if mh.scores[idx] > best_score:
                best_state = mh.states[idx]
                best_score = mh.scores[idx]

        return {
            "final_states": mh.states,
            "best_score": best_score,
            "best_state": best_state,
            "step_best_scores": step_best_scores,
            "step_best_state_size": step_best_state_size,
        }

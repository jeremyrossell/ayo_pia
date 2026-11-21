"""
Low-level TSP heuristics adapted from my homework 02 for the [0, 1]^2 unit square.

Algorithms included:
    1. 2-opt Local Search (Hill Climbing) starting from a random permutation.
    2. Simulated Annealing (SA) with 2-opt neighborhood and exponential cooling.
    3. MST 2-Approximation (Prim + Preorder Walk) followed by 2-opt post-processing.
"""

import math
import random

# Master seed (my student ID)
BASE_SEED: int = 2173396

# Default Simulated Annealing parameters rescaled by 1/100 for [0, 1]^2 coordinates
DEFAULT_SA_STEPS: int = 15_000
DEFAULT_SA_T_START: float = 2.0
DEFAULT_SA_T_END: float = 1e-4


def compute_dist_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    n = len(points)
    return [[math.dist(points[i], points[j]) for j in range(n)] for i in range(n)]


def tour_length(tour: list[int], dist_matrix: list[list[float]]) -> float:
    n = len(tour)
    return sum(dist_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))


# Applies first-improvement 2-opt hill climbing until reaching a local optimum.
def two_opt_local_search(
    initial_tour: list[int],                # starting permutation of city indices
    dist_matrix: list[list[float]],   # precomputed pairwise Euclidean distance matrix
) -> tuple[list[int], float]:

    tour = list(initial_tour)
    n = len(tour)
    if n < 4:
        return tour, tour_length(tour, dist_matrix)

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b = tour[i - 1], tour[i]
                c, d = tour[k], tour[(k + 1) % n]

                # O(1) cost delta when reversing subsegment tour[i : k + 1]
                delta = (dist_matrix[a][c] + dist_matrix[b][d]) - (
                    dist_matrix[a][b] + dist_matrix[c][d]
                )

                if delta < -1e-12:
                    tour[i : k + 1] = reversed(tour[i : k + 1])
                    improved = True
                    break
            if improved:
                break

    return tour, tour_length(tour, dist_matrix)


# Executes Simulated Annealing with a random 2-opt neighborhood and exponential cooling.
def simulated_annealing_2opt(
    initial_tour: list[int],                     # starting permutation of city indices
    dist_matrix: list[list[float]],         # precomputed pairwise Euclidean distance matrix
    steps: int = DEFAULT_SA_STEPS,               # total number of annealing iterations
    t_start: float = DEFAULT_SA_T_START,    # initial temperature scaled for [0, 1]^2
    t_end: float = DEFAULT_SA_T_END,        # final temperature threshold
    seed: int = BASE_SEED,                       # deterministic RNG seed for Metropolis acceptance and move sampling
) -> tuple[list[int], float]:

    rng = random.Random(seed)
    current_tour = list(initial_tour)
    n = len(current_tour)
    current_cost = tour_length(current_tour, dist_matrix)

    if n < 4 or steps <= 0:
        return current_tour, current_cost

    best_tour = list(current_tour)
    best_cost = current_cost

    alpha = (t_end / t_start) ** (1.0 / steps)
    temperature = t_start

    for _ in range(steps):
        if temperature <= 1e-12:
            break

        i, k = sorted(rng.sample(range(n), 2))
        # Skip full-tour reversal (equivalent to the same cyclic tour)
        if i == 0 and k == n - 1:
            temperature *= alpha
            continue

        a, b = current_tour[i - 1], current_tour[i]
        c, d = current_tour[k], current_tour[(k + 1) % n]
        delta = (dist_matrix[a][c] + dist_matrix[b][d]) - (
            dist_matrix[a][b] + dist_matrix[c][d]
        )

        if delta < 0.0 or rng.random() < math.exp(-delta / temperature):
            current_tour[i : k + 1] = reversed(current_tour[i : k + 1])
            current_cost += delta
            if current_cost < best_cost:
                best_tour = list(current_tour)
                best_cost = current_cost

        temperature *= alpha

    # Recompute exact cost from scratch to eliminate cumulative floating-point drift
    return best_tour, tour_length(best_tour, dist_matrix)


# Builds a Minimum Spanning Tree via Prim's algorithm rooted at city 0.
def mst_prim(num_cities: int, dist_matrix: list[list[float]]) -> dict[int, list[int]]:
    best = {v: (dist_matrix[0][v], 0) for v in range(1, num_cities)}
    children: dict[int, list[int]] = {v: [] for v in range(num_cities)}

    while best:
        v = min(best, key=lambda u: best[u][0])
        parent = best.pop(v)[1]
        children[parent].append(v)
        for u in best:
            d = dist_matrix[v][u]
            if d < best[u][0]:
                best[u] = (d, v)

    return children


# Performs a depth-first preorder traversal of the MST, shortcutting repeated visits.
def preorder_walk(children: dict[int, list[int]], root: int = 0) -> list[int]:
    return [root] + [
        city for child in children[root] for city in preorder_walk(children, child)
    ]


# Constructs an MST 2-approximation tour and refines it with 2-opt local search.
def mst_2opt(dist_matrix: list[list[float]]) -> tuple[list[int], float]:
    num_cities = len(dist_matrix)
    children = mst_prim(num_cities, dist_matrix)
    mst_tour = preorder_walk(children, root=0)
    return two_opt_local_search(mst_tour, dist_matrix)


# Runs all three TSP solvers on a single instance and identifies the winning algorithm.
#    Both 2-opt hill climbing and Simulated Annealing start from the exact same random
#    initial tour generated from `seed` to ensure a fair paired comparison.
def solve_instance(
    points: list[tuple[float, float]],
    seed: int = BASE_SEED,
    sa_steps: int = DEFAULT_SA_STEPS,
    sa_t_start: float = DEFAULT_SA_T_START,
    sa_t_end: float = DEFAULT_SA_T_END,
) -> dict[str, float | str]:                    # three costs and `best_algorithm` label

    dist_matrix = compute_dist_matrix(points)
    num_cities = len(points)

    rng = random.Random(seed)
    initial_tour = list(range(num_cities))
    rng.shuffle(initial_tour)

    _, cost_2opt = two_opt_local_search(initial_tour, dist_matrix)
    _, cost_sa = simulated_annealing_2opt(
        initial_tour,
        dist_matrix,
        steps=sa_steps,
        t_start=sa_t_start,
        t_end=sa_t_end,
        seed=seed,
    )
    _, cost_mst = mst_2opt(dist_matrix)

    costs: dict[str, float] = {
        "2opt_hill_climbing": cost_2opt,
        "sa_2opt": cost_sa,
        "mst_2opt": cost_mst,
    }
    best_algorithm = min(costs, key=lambda name: costs[name])

    return {
        "cost_2opt_hill_climbing": cost_2opt,
        "cost_sa_2opt": cost_sa,
        "cost_mst_2opt": cost_mst,
        "best_algorithm": best_algorithm,
    }

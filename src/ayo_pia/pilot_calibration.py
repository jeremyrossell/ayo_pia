"""
Pilot calibration script for TSP instance labeling.

Runs a stratified sample of 27 instances (one per (n, distribution) pair from train.csv)
to verify CSV parsing, measure runtime scalability, and inspect class balance before
running the full 500-instance labeling pipeline.
"""

import csv
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from tsp_solvers import (
    BASE_SEED,
    DEFAULT_SA_STEPS,
    DEFAULT_SA_T_END,
    DEFAULT_SA_T_START,
    solve_instance,
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
INSTANCES_DIR = DATA_DIR / "instances"

# Pilot SA parameters
PILOT_SA_STEPS: int = DEFAULT_SA_STEPS
PILOT_SA_T_START: float = DEFAULT_SA_T_START
PILOT_SA_T_END: float = DEFAULT_SA_T_END


def derive_instance_seed(instance_id: str, base_seed: int = BASE_SEED) -> int:
    # deterministic
    numeric_suffix = int(instance_id.split("_")[-1])
    return base_seed + numeric_suffix


def load_instance_points(instance_id: str) -> list[tuple[float, float]]:

    filepath = INSTANCES_DIR / f"{instance_id}.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Instance file not found: {filepath}")

    points: list[tuple[float, float]] = []
    with filepath.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                x, y = float(row[0]), float(row[1])
                points.append((x, y))
            except ValueError:
                # skip header row
                continue

    return points


def select_stratified_pilot(train_csv_path: Path) -> list[dict[str, str]]:
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training metadata file not found: {train_csv_path}")

    seen_strata: set[tuple[int, str]] = set()
    pilot_rows: list[dict[str, str]] = []

    with train_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stratum = (int(row["n"]), row["distribution"])
            if stratum not in seen_strata:
                seen_strata.add(stratum)
                pilot_rows.append(row)

    pilot_rows.sort(key=lambda r: (int(r["n"]), r["distribution"]))
    return pilot_rows


def run_pilot() -> None:
    pilot_rows = select_stratified_pilot(TRAIN_CSV)
    total_pilot = len(pilot_rows)

    print("=" * 82)
    print("PILOT CALIBRATION — STRATIFIED TSP BENCHMARK (1 INSTANCE PER (n, DISTRIBUTION))")
    print("=" * 82)
    print(f"Project Root    : {PROJECT_ROOT}")
    print(f"Base Seed       : {BASE_SEED}")
    print(
        f"SA Parameters   : steps={PILOT_SA_STEPS:,}, "
        f"T_start={PILOT_SA_T_START}, T_end={PILOT_SA_T_END}"
    )
    print(f"Pilot Sample    : {total_pilot} instances\n")

    print(
        f"{'Instance':<12}{'n':>4}  {'Distribution':<11}"
        f"{'2-opt':>10}{'SA+2opt':>10}{'MST+2opt':>10}  {'Winner':<20}{'Time(s)':>8}"
    )
    print("-" * 89)

    win_counts: Counter[str] = Counter()
    wins_by_dist: dict[str, Counter[str]] = defaultdict(Counter)
    times_by_n: dict[int, list[float]] = defaultdict(list)

    t_pilot_start = time.perf_counter()

    for row in pilot_rows:
        inst_id = row["instance_id"]
        n = int(row["n"])
        dist_type = row["distribution"]

        points = load_instance_points(inst_id)
        if len(points) != n:
            raise ValueError(f"{inst_id}: expected {n} points, found {len(points)}")

        seed = derive_instance_seed(inst_id, BASE_SEED)

        t0 = time.perf_counter()
        result = solve_instance(
            points=points,
            seed=seed,
            sa_steps=PILOT_SA_STEPS,
            sa_t_start=PILOT_SA_T_START,
            sa_t_end=PILOT_SA_T_END,
        )
        elapsed = time.perf_counter() - t0

        winner = str(result["best_algorithm"])
        win_counts[winner] += 1
        wins_by_dist[dist_type][winner] += 1
        times_by_n[n].append(elapsed)

        print(
            f"{inst_id:<12}{n:>4}  {dist_type:<11}"
            f"{result['cost_2opt_hill_climbing']:>10.4f}"
            f"{result['cost_sa_2opt']:>10.4f}"
            f"{result['cost_mst_2opt']:>10.4f}  "
            f"{winner:<20}{elapsed:>8.4f}"
        )

    total_elapsed = time.perf_counter() - t_pilot_start
    avg_time_per_instance = total_elapsed / total_pilot
    est_seq_500 = avg_time_per_instance * 500
    cpu_cores = os.cpu_count() or 1
    est_par_500 = est_seq_500 / max(1, min(cpu_cores, 8))

    print("-" * 89)
    print("\n--- Pilot Class Balance Summary ---")
    for algo in ("2opt_hill_climbing", "mst_2opt", "sa_2opt"):
        count = win_counts[algo]
        pct = (count / total_pilot) * 100.0
        print(f"  {algo:<22}: {count:>2d} / {total_pilot} ({pct:5.1f}%)")

    print("\n--- Wins by Spatial Distribution ---")
    for dist_type in sorted(wins_by_dist.keys()):
        counts_str = ", ".join(
            f"{algo}={wins_by_dist[dist_type][algo]}"
            for algo in ("2opt_hill_climbing", "mst_2opt", "sa_2opt")
        )
        print(f"  {dist_type:<12}: {counts_str}")

    print("\n--- Runtime Scalability & Full Dataset Projection (500 Instances) ---")
    for n in sorted(times_by_n.keys()):
        mean_t = sum(times_by_n[n]) / len(times_by_n[n])
        print(f"  n = {n:>3d} : avg {mean_t:.4f} s / instance")

    print(f"\n  Total pilot time ({total_pilot} instances) : {total_elapsed:.2f} s")
    print(f"  Estimated sequential (500 instances) : {est_seq_500:.1f} s (~{est_seq_500 / 60:.2f} min)")
    print(
        f"  Estimated parallel ({cpu_cores} logical CPUs) : "
        f"~{est_par_500:.1f} s (~{est_par_500 / 60:.2f} min)"
    )


if __name__ == "__main__":
    run_pilot()

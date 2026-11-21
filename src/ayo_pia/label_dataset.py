"""
Full dataset labeling pipeline for Part 1 (Selection Hyper-Heuristic for TSP).

Executes all three low-level heuristics (2-opt hill climbing, Simulated Annealing + 2-opt,
and MST 2-approximation + 2-opt) across all instances in data/train.csv and
data/test.csv, appending the tour costs and `best_algorithm` label to:
    - output/train_labeled.csv
    - output/test_labeled.csv
"""

import csv
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    from ayo_pia.tsp_solvers import (
        BASE_SEED,
        DEFAULT_SA_STEPS,
        DEFAULT_SA_T_END,
        DEFAULT_SA_T_START,
        solve_instance,
    )
except ImportError:
    from tsp_solvers import (
        BASE_SEED,
        DEFAULT_SA_STEPS,
        DEFAULT_SA_T_END,
        DEFAULT_SA_T_START,
        solve_instance,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INSTANCES_DIR = DATA_DIR / "instances"
TRAIN_INPUT_CSV = DATA_DIR / "train.csv"
TEST_INPUT_CSV = DATA_DIR / "test.csv"

OUTPUT_DIR = PROJECT_ROOT / "output"
TRAIN_OUTPUT_CSV = OUTPUT_DIR / "train_labeled.csv"
TEST_OUTPUT_CSV = OUTPUT_DIR / "test_labeled.csv"

LABEL_COLUMNS: list[str] = [
    "cost_2opt_hill_climbing",
    "cost_sa_2opt",
    "cost_mst_2opt",
    "best_algorithm",
]


def derive_instance_seed(instance_id: str, base_seed: int = BASE_SEED) -> int:
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
                points.append((float(row[0]), float(row[1])))
            except ValueError:
                # skip header row
                continue

    return points


# Worker function that solves a single TSP instance and appends cost/label columns
def _process_single_row(row: dict[str, str]) -> dict[str, object]:
    instance_id = row["instance_id"]
    expected_n = int(row["n"])

    points = load_instance_points(instance_id)
    if len(points) != expected_n:
        raise ValueError(
            f"{instance_id}: expected {expected_n} coordinates, found {len(points)}"
        )

    seed = derive_instance_seed(instance_id, BASE_SEED)
    metrics = solve_instance(
        points=points,
        seed=seed,
        sa_steps=DEFAULT_SA_STEPS,
        sa_t_start=DEFAULT_SA_T_START,
        sa_t_end=DEFAULT_SA_T_END,
    )

    labeled_row: dict[str, object] = dict(row)
    labeled_row.update(metrics)
    return labeled_row


# Reads an unlabeled split CSV, runs all solvers in parallel, and writes the labeled CSV
def label_split(
    input_csv: Path,
    output_csv: Path,
    max_workers: int,
) -> list[dict[str, object]]:

    if not input_csv.exists():
        raise FileNotFoundError(f"Input dataset split not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV header in {input_csv}")
        base_fieldnames = list(reader.fieldnames)
        rows = list(reader)

    output_fieldnames = base_fieldnames + LABEL_COLUMNS

    # ProcessPoolExecutor.map preserves the exact input row ordering
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        labeled_rows = list(executor.map(_process_single_row, rows, chunksize=8))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(labeled_rows)

    return labeled_rows


def print_class_balance(title: str, rows: list[dict[str, object]]) -> None:
    total = len(rows)
    counts = Counter(str(r["best_algorithm"]) for r in rows)
    print(f"\n--- {title} (N = {total}) ---")
    for algo in ("2opt_hill_climbing", "mst_2opt", "sa_2opt"):
        c = counts[algo]
        pct = (c / total) * 100.0 if total > 0 else 0.0
        print(f"  {algo:<22}: {c:>3d} ({pct:5.1f}%)")


def print_stratified_breakdown(rows: list[dict[str, object]]) -> None:
    by_dist: dict[str, Counter[str]] = defaultdict(Counter)
    by_n: dict[int, Counter[str]] = defaultdict(Counter)

    for r in rows:
        winner = str(r["best_algorithm"])
        by_dist[str(r["distribution"])][winner] += 1
        by_n[int(str(r["n"]))][winner] += 1

    print("\n--- Overall Wins by Spatial Distribution (All 500 Instances) ---")
    for dist_name in sorted(by_dist.keys()):
        c = by_dist[dist_name]
        print(
            f"  {dist_name:<12}: "
            f"2opt={c['2opt_hill_climbing']:>3d}, "
            f"mst_2opt={c['mst_2opt']:>3d}, "
            f"sa_2opt={c['sa_2opt']:>3d}"
        )

    print("\n--- Overall Wins by Problem Size n (All 500 Instances) ---")
    for n_val in sorted(by_n.keys()):
        c = by_n[n_val]
        print(
            f"  n = {n_val:>3d}     : "
            f"2opt={c['2opt_hill_climbing']:>3d}, "
            f"mst_2opt={c['mst_2opt']:>3d}, "
            f"sa_2opt={c['sa_2opt']:>3d}"
        )


def main() -> None:
    workers = os.cpu_count() or 1
    print("=" * 80)
    print("FULL TSP DATASET LABELING PIPELINE")
    print("=" * 80)
    print(f"Base Seed     : {BASE_SEED}")
    print(
        f"SA Parameters : steps={DEFAULT_SA_STEPS:,}, "
        f"T_start={DEFAULT_SA_T_START}, T_end={DEFAULT_SA_T_END}"
    )
    print(f"Worker CPUs   : {workers}\n")

    t0 = time.perf_counter()

    print(f"Labeling training set ({TRAIN_INPUT_CSV.name})...")
    train_labeled = label_split(TRAIN_INPUT_CSV, TRAIN_OUTPUT_CSV, max_workers=workers)
    print(f"  -> Saved {len(train_labeled)} rows to {TRAIN_OUTPUT_CSV}")

    print(f"Labeling test set ({TEST_INPUT_CSV.name})...")
    test_labeled = label_split(TEST_INPUT_CSV, TEST_OUTPUT_CSV, max_workers=workers)
    print(f"  -> Saved {len(test_labeled)} rows to {TEST_OUTPUT_CSV}")

    elapsed = time.perf_counter() - t0
    all_labeled = train_labeled + test_labeled

    print_class_balance("Train Set Class Balance", train_labeled)
    print_class_balance("Test Set Class Balance", test_labeled)
    print_class_balance("Total Dataset Class Balance", all_labeled)
    print_stratified_breakdown(all_labeled)

    print("\n" + "=" * 80)
    print(f"COMPLETED IN {elapsed:.2f} SECONDS")
    print("=" * 80)


if __name__ == "__main__":
    main()

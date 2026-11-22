"""Evaluation outputs: confusion matrices, per-instance regret and model comparison.

Reuses constants and compute_regret from ml_pipeline.py. Models are refit with the
best hyperparameters found by the grid search in ml_pipeline.py (no new grid search).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
try:
    from ayo_pia.ml_pipeline import (
        CATEGORICAL_COLS,
        COST_MAP,
        FEATURE_COLS,
        NUMERIC_COLS,
        TARGET_COL,
        compute_regret,
    )
except ImportError:
    from ml_pipeline import (
        CATEGORICAL_COLS,
        COST_MAP,
        FEATURE_COLS,
        NUMERIC_COLS,
        TARGET_COL,
        compute_regret,
    )
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
LABELS = sorted(COST_MAP)


def build_models() -> dict:
    tree = Pipeline([
        ("prep", ColumnTransformer([
            ("num", "passthrough", NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ])),
        ("clf", DecisionTreeClassifier(
            criterion="gini", max_depth=6, min_samples_leaf=10,
            class_weight=None, random_state=42,
        )),
    ])
    knn = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ])),
        ("clf", KNeighborsClassifier(
            n_neighbors=15, weights="uniform", metric="euclidean",
        )),
    ])
    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "decision_tree": tree,
        "knn": knn,
    }


def per_instance_regret(test_df: pd.DataFrame, preds) -> pd.DataFrame:
    best_cost = test_df[list(COST_MAP.values())].min(axis=1)
    chosen_cost = pd.Series(
        [row[COST_MAP[p]] for p, (_, row) in zip(preds, test_df.iterrows())],
        index=test_df.index,
    )
    return pd.DataFrame({
        "instance_id": test_df["instance_id"],
        "n": test_df["n"],
        "distribution": test_df["distribution"],
        "real_best": test_df[TARGET_COL],
        "predicted": list(preds),
        "cost_chosen": chosen_cost,
        "cost_best": best_cost,
        "regret": chosen_cost - best_cost,
    })


def save_confusion_matrix(y_true, preds, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, preds, labels=LABELS, ax=ax, cmap="Blues", xticks_rotation=30,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_tree(model: Pipeline, path: Path) -> None:
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    names = list(prep.get_feature_names_out())

    fig, ax = plt.subplots(figsize=(40, 16))
    plot_tree(
        clf, feature_names=names, class_names=list(clf.classes_),
        filled=True, rounded=True, fontsize=7, ax=ax,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(OUTPUT_DIR / "train_labeled.csv")
    test_df = pd.read_csv(OUTPUT_DIR / "test_labeled.csv")
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    summary = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        if name == "decision_tree":
            save_tree(model, OUTPUT_DIR / "decision_tree.png")
        preds = model.predict(X_test)

        save_confusion_matrix(
            y_test, preds, f"Confusion matrix (test): {name}",
            OUTPUT_DIR / f"confusion_{name}.png",
        )
        per_instance_regret(test_df, preds).to_csv(
            OUTPUT_DIR / f"regret_per_instance_{name}.csv", index=False,
        )

        regret = compute_regret(test_df, preds)
        summary.append({
            "model": name,
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "regret_mean": round(regret["mean"], 6),
            "regret_median": round(regret["median"], 6),
            "regret_rel_pct_mean": round(regret["rel_pct_mean"], 2),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
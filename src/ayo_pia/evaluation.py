"""Evaluation outputs: confusion matrices, per-instance regret, model comparison and figures.

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
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
LABELS = sorted(COST_MAP)
ALGO_COLORS = {
    "2opt_hill_climbing": "#4C72B0",
    "mst_2opt": "#DD8452",
    "sa_2opt": "#55A868",
}


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


def save_regret_comparison(path: Path) -> None:
    summary = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
    models = summary["model"].tolist()
    n_regret = [
        int((pd.read_csv(OUTPUT_DIR / f"regret_per_instance_{m}.csv")["regret"] > 1e-12).sum())
        for m in models
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bars = axes[0].bar(models, summary["regret_rel_pct_mean"], color="#4C72B0")
    axes[0].bar_label(bars, fmt="%.2f%%")
    axes[0].set_title("Regret relativo promedio (%)")

    bars = axes[1].bar(models, n_regret, color="#C44E52")
    axes[1].bar_label(bars)
    axes[1].set_title("Instancias con regret > 0 (de 149)")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_wins_by_size_and_dist(path: Path) -> None:
    train_df = pd.read_csv(OUTPUT_DIR / "train_labeled.csv")
    test_df = pd.read_csv(OUTPUT_DIR / "test_labeled.csv")
    all_df = pd.concat([train_df, test_df], ignore_index=True)

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2, 1]},
    )
    for ax, col, title in [
        (axes[0], "n", "Victorias por tamaño n"),
        (axes[1], "distribution", "Victorias por distribución"),
    ]:
        counts = pd.crosstab(all_df[col], all_df[TARGET_COL])[LABELS]
        counts.plot(
            kind="bar", stacked=True, ax=ax, rot=0,
            color=[ALGO_COLORS[a] for a in LABELS],
        )
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Instancias (N=500)")
    axes[0].legend(title="best_algorithm")
    axes[1].get_legend().remove()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_feature_importance(tree_model, knn_model, X_test, y_test, path: Path) -> None:
    # Gini del árbol; los indicadores one-hot de distribution se suman en una barra
    names = tree_model.named_steps["prep"].get_feature_names_out()
    gini = pd.Series(tree_model.named_steps["clf"].feature_importances_, index=names)
    gini.index = [i.split("__", 1)[1] for i in gini.index]
    gini.index = [
        "distribution" if i.startswith("distribution_") else i for i in gini.index
    ]
    gini = gini.groupby(level=0).sum().reindex(FEATURE_COLS) * 100

    # Importancia por permutación (accuracy) del KNN sobre el conjunto de prueba
    perm = permutation_importance(
        knn_model, X_test, y_test, n_repeats=30, random_state=42, scoring="accuracy",
    )
    perm_pp = pd.Series(perm.importances_mean, index=FEATURE_COLS) * 100
    perm_sd = pd.Series(perm.importances_std, index=FEATURE_COLS) * 100

    order = gini.sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    gini.reindex(order).plot(kind="barh", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Importancia Gini (Árbol de Decisión), %")
    perm_pp.reindex(order).plot(
        kind="barh", ax=axes[1], color="#55A868", xerr=perm_sd.reindex(order),
    )
    axes[1].set_title("Importancia por permutación (KNN), pp de accuracy")
    axes[1].axvline(0, color="black", lw=0.8)
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
    fitted = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        fitted[name] = model

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

    save_regret_comparison(OUTPUT_DIR / "regret_comparison.png")
    save_wins_by_size_and_dist(OUTPUT_DIR / "wins_by_size_and_dist.png")
    save_feature_importance(
        fitted["decision_tree"], fitted["knn"], X_test, y_test,
        OUTPUT_DIR / "feature_importance.png",
    )


if __name__ == "__main__":
    main()
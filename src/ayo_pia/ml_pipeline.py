import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

NUMERIC_COLS = [
    "n",
    "mean_pairwise_dist",
    "std_coords",
    "mean_centroid_dist",
    "mean_nn_dist",
]
CATEGORICAL_COLS = ["distribution"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
TARGET_COL = "best_algorithm"

COST_MAP = {
    "2opt_hill_climbing": "cost_2opt_hill_climbing",
    "sa_2opt": "cost_sa_2opt",
    "mst_2opt": "cost_mst_2opt",
}


def print_class_balance(name: str, series: pd.Series) -> None:
    counts = series.value_counts()
    pcts = (series.value_counts(normalize=True) * 100.0).round(2)
    summary = pd.DataFrame({"count": counts, "pct": pcts})
    print(f"\n--- Balance de clases: {name} (n={len(series)}) ---")
    print(summary.to_string())


def compute_regret(test_df: pd.DataFrame, preds) -> dict[str, float]:
    best_real = test_df[list(COST_MAP.values())].min(axis=1)
    chosen = [row[COST_MAP[p]] for p, (_, row) in zip(preds, test_df.iterrows())]
    regret = pd.Series(chosen, index=test_df.index) - best_real
    rel_pct = np.where(best_real > 0, (regret / best_real) * 100.0, 0.0)
    return {
        "mean": float(regret.mean()),
        "median": float(regret.median()),
        "rel_pct_mean": float(np.mean(rel_pct)),
    }


def evaluate_model(
    name: str,
    y_test: pd.Series,
    preds,
    labels: list[str],
    regret_stats: dict[str, float],
    best_params: dict | None = None,
    cv_score: float | None = None,
) -> None:
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"real_{lbl}" for lbl in labels],
        columns=[f"pred_{lbl}" for lbl in labels],
    )

    print(f"\n=== {name} ===")
    if cv_score is not None:
        print(f"Best CV Accuracy (5-fold): {cv_score:.4f}")
    if best_params is not None:
        print(f"Best Params: {best_params}")
    print(f"Test Accuracy: {acc:.4f}")
    print(
        f"Regret -> mean: {regret_stats['mean']:.6f} | "
        f"median: {regret_stats['median']:.6f} | "
        f"rel_mean: {regret_stats['rel_pct_mean']:.2f}%"
    )
    print("Matriz de confusion (test):")
    print(cm_df.to_string())
    print("Reporte de clasificacion (test):")
    print(classification_report(y_test, preds, labels=labels, digits=4, zero_division=0))


def main() -> None:
    train_df = pd.read_csv("output/train_labeled.csv")
    test_df = pd.read_csv("output/test_labeled.csv")

    print_class_balance("train", train_df[TARGET_COL])
    print_class_balance("test", test_df[TARGET_COL])

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Baseline (most_frequent)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base_preds = baseline.predict(X_test)
    labels = sorted(list(set(y_train.unique()) | set(y_test.unique())))
    evaluate_model(
        name="Baseline (most_frequent)",
        y_test=y_test,
        preds=base_preds,
        labels=labels,
        regret_stats=compute_regret(test_df, base_preds),
    )

    # 2. Decision Tree
    dt_prep = ColumnTransformer([
        ("num", "passthrough", NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])
    dt_pipe = Pipeline([
        ("prep", dt_prep),
        ("clf", DecisionTreeClassifier(random_state=42)),
    ])
    dt_grid = GridSearchCV(
        dt_pipe,
        {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [2, 3, 4, 5, 6, 8, None],
            "clf__min_samples_leaf": [1, 2, 5, 10],
            "clf__class_weight": [None, "balanced"],
        },
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    ).fit(X_train, y_train)
    dt_preds = dt_grid.predict(X_test)
    evaluate_model(
        name="Decision Tree",
        y_test=y_test,
        preds=dt_preds,
        labels=list(dt_grid.classes_),
        regret_stats=compute_regret(test_df, dt_preds),
        best_params=dt_grid.best_params_,
        cv_score=float(dt_grid.best_score_),
    )

    # 3. KNN
    knn_prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])
    knn_pipe = Pipeline([
        ("prep", knn_prep),
        ("clf", KNeighborsClassifier()),
    ])
    knn_grid = GridSearchCV(
        knn_pipe,
        {
            "clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        },
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    ).fit(X_train, y_train)
    knn_preds = knn_grid.predict(X_test)
    evaluate_model(
        name="KNN",
        y_test=y_test,
        preds=knn_preds,
        labels=list(knn_grid.classes_),
        regret_stats=compute_regret(test_df, knn_preds),
        best_params=knn_grid.best_params_,
        cv_score=float(knn_grid.best_score_),
    )


if __name__ == "__main__":
    main()
"""Leakage-safe model comparison, tuning, evaluation, and artifact creation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".matplotlib_cache"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .data import DEFAULT_DATA_PATH, PROJECT_ROOT, feature_schema, load_and_clean_data


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


class BalancedGradientBoostingClassifier(ClassifierMixin, BaseEstimator):
    """Gradient boosting wrapper that supplies balanced sample weights."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        n_estimators: int = 120,
        max_depth: int = 2,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = GradientBoostingClassifier(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        weights = compute_sample_weight(class_weight="balanced", y=y)
        self.model_.fit(X, y, sample_weight=weights)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build imputers, scaling, and one-hot encoding to be fit within CV folds."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ],
        remainder="drop",
    )


def make_models() -> Dict[str, BaseEstimator]:
    """Return the requested model families with reproducible seeds."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Support Vector Machine": SVC(
            C=1.0,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": BalancedGradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def make_pipeline(X: pd.DataFrame, estimator: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[("preprocessor", make_preprocessor(X)), ("classifier", clone(estimator))]
    )


def _search_space(model_name: str) -> dict:
    spaces = {
        "Logistic Regression": {"classifier__C": [0.1, 0.3, 1.0, 3.0, 10.0]},
        "Decision Tree": {
            "classifier__max_depth": [3, 4, 6, 8, None],
            "classifier__min_samples_leaf": [5, 10, 20, 30],
        },
        "Random Forest": {
            "classifier__n_estimators": [150, 250, 350],
            "classifier__max_depth": [None, 6, 10, 14],
            "classifier__min_samples_leaf": [2, 3, 5, 10],
        },
        "Support Vector Machine": {
            "classifier__C": [0.3, 1.0, 3.0, 10.0],
            "classifier__gamma": ["scale", "auto"],
        },
        "Gradient Boosting": {
            "classifier__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "classifier__n_estimators": [80, 120, 180],
            "classifier__max_depth": [1, 2, 3],
        },
    }
    return spaces[model_name]


def _positive_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classes = model.named_steps["classifier"].classes_
    positive_index = int(np.flatnonzero(classes == 1)[0])
    return model.predict_proba(X)[:, positive_index]


def _choose_f2_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose a recall-forward threshold using out-of-fold training predictions."""
    thresholds = np.linspace(0.10, 0.90, 81)
    scores = [fbeta_score(y_true, probabilities >= threshold, beta=2, zero_division=0) for threshold in thresholds]
    best = np.flatnonzero(np.isclose(scores, np.max(scores)))
    # If F2 ties, select the higher threshold to avoid needless false alarms.
    return float(thresholds[best[-1]])


def _save_evaluation_plots(y_true, probabilities, predictions, output_dir: Path) -> None:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(title="Test-set confusion matrix", xlabel="Predicted label", ylabel="Actual label")
    ax.set_xticks([0, 1], ["Stayed", "Churned"])
    ax.set_yticks([0, 1], ["Stayed", "Churned"])
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, probabilities):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set(title="ROC curve (held-out test set)", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"Average precision = {average_precision_score(y_true, probabilities):.3f}")
    ax.set(title="Precision–recall curve (held-out test set)", xlabel="Recall", ylabel="Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close(fig)


def run_training(
    data_path: str | Path = DEFAULT_DATA_PATH,
    project_root: str | Path = PROJECT_ROOT,
    cv_folds: int = CV_FOLDS,
) -> dict:
    """Compare models, tune the top CV-F1 model, evaluate once on test data."""
    root = Path(project_root)
    models_dir = root / "models"
    outputs_dir = root / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    X, y, cleaned = load_and_clean_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    print(f"Loaded {len(cleaned):,} real IBM sample rows; churn rate = {y.mean():.1%}.")
    print("Comparing models with stratified cross-validation on the training split...")
    model_rows = []
    candidates = {}
    for name, estimator in make_models().items():
        pipeline = make_pipeline(X_train, estimator)
        result = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise"
        )
        candidates[name] = pipeline
        # Keep a fitted, untuned comparator for hands-on inspection and reuse.
        fitted_comparator = clone(pipeline).fit(X_train, y_train)
        model_slug = name.lower().replace(" ", "_")
        joblib.dump(
            {
                "model": fitted_comparator,
                "model_name": name,
                "feature_columns": X.columns.tolist(),
                "decision_threshold": 0.5,
            },
            models_dir / f"{model_slug}.joblib",
        )
        row = {"model": name}
        for metric in scoring:
            values = result[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        model_rows.append(row)
        print(f"  {name}: F1={row['f1_mean']:.3f} ± {row['f1_std']:.3f}; recall={row['recall_mean']:.3f}")

    comparison = pd.DataFrame(model_rows).sort_values(
        ["f1_mean", "roc_auc_mean"], ascending=False
    ).reset_index(drop=True)
    selected_name = str(comparison.loc[0, "model"])
    print(f"Tuning CV-F1 leader: {selected_name}...")
    search = RandomizedSearchCV(
        estimator=candidates[selected_name],
        param_distributions=_search_space(selected_name),
        n_iter=8,
        scoring="f1",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    selected_model = search.best_estimator_
    print(f"Best tuning CV F1: {search.best_score_:.3f}")

    threshold_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    oof_probabilities = cross_val_predict(
        selected_model,
        X_train,
        y_train,
        cv=threshold_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold = _choose_f2_threshold(y_train, oof_probabilities)
    selected_model.fit(X_train, y_train)

    test_probabilities = _positive_probability(selected_model, X_test)
    test_predictions = (test_probabilities >= threshold).astype(int)
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
        "f2": float(fbeta_score(y_test, test_predictions, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "average_precision": float(average_precision_score(y_test, test_probabilities)),
        "decision_threshold": threshold,
    }
    report = classification_report(
        y_test, test_predictions, target_names=["Stayed", "Churned"], output_dict=True, zero_division=0
    )
    comparison.to_csv(outputs_dir / "model_comparison.csv", index=False)
    _save_evaluation_plots(y_test, test_probabilities, test_predictions, outputs_dir)

    print("Calculating held-out permutation importance...")
    importance = permutation_importance(
        selected_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(outputs_dir / "feature_importance.csv", index=False)
    top = importance_df.head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#2364AA")
    ax.set(title="Permutation importance on held-out test data", xlabel="Mean ROC-AUC decrease when shuffled")
    fig.tight_layout()
    fig.savefig(outputs_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

    bundle = {
        "model": selected_model,
        "model_name": selected_name,
        "decision_threshold": threshold,
        "feature_columns": X.columns.tolist(),
        "feature_schema": feature_schema(X_train),
        "reference_values": X_train.median(numeric_only=True).to_dict()
        | X_train.select_dtypes(exclude=["number"]).mode(dropna=True).iloc[0].to_dict(),
        "random_state": RANDOM_STATE,
    }
    joblib.dump(bundle, models_dir / "churn_model.joblib")
    with (outputs_dir / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": "IBM Telco Customer Churn sample",
                "rows_after_cleaning": int(len(cleaned)),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "churn_rate": float(y.mean()),
                "selected_model": selected_name,
                "best_parameters": search.best_params_,
                "test_metrics": test_metrics,
                "classification_report": report,
                "note": "Held-out metrics describe this public sample, not expected real-world business performance.",
            },
            file,
            indent=2,
        )
    print("Held-out test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.3f}")
    print(f"Saved model and reports under {root}.")
    return {"comparison": comparison, "metrics": test_metrics, "selected_model": selected_name}

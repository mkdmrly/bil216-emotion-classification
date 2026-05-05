from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class ModelChoice:
    name: str
    estimator: object
    param_grid: dict[str, list]


def get_model_choices() -> list[ModelChoice]:
    return [
        ModelChoice(
            name="svm",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVC()),
                ]
            ),
            param_grid={
                "model__C": [1, 10, 50],
                "model__gamma": ["scale", 0.01, 0.001],
                "model__kernel": ["rbf"],
            },
        ),
        ModelChoice(
            name="random_forest",
            estimator=RandomForestClassifier(random_state=42),
            param_grid={
                "n_estimators": [200, 400],
                "max_depth": [None, 20, 40],
                "min_samples_leaf": [1, 2],
            },
        ),
        ModelChoice(
            name="knn",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            param_grid={
                "model__n_neighbors": [3, 5, 7],
                "model__weights": ["uniform", "distance"],
            },
        ),
        ModelChoice(
            name="logistic_regression",
            estimator=Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            multi_class="auto",
                        ),
                    ),
                ]
            ),
            param_grid={
                "model__C": [0.1, 1.0, 10.0],
            },
        ),
    ]


def search_best_model(X_train, y_train, model_name: str = "auto", cv: int = 5):
    choices = get_model_choices()
    if model_name != "auto":
        choices = [choice for choice in choices if choice.name == model_name]
        if not choices:
            available = ", ".join(choice.name for choice in get_model_choices())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    best_search = None
    best_score = float("-inf")
    for choice in choices:
        search = GridSearchCV(
            estimator=choice.estimator,
            param_grid=choice.param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        if search.best_score_ > best_score:
            best_search = search
            best_score = search.best_score_

    if best_search is None:
        raise RuntimeError("Model search failed.")

    return best_search


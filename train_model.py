"""Reproduce the champion model (Random Forest, tuned with GridSearchCV + stratified 5-fold CV).

Usage:  python scripts/train_model.py
Writes: model/champion_model.joblib, model/metadata.json
"""
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

SEED = 42
OUT = Path(__file__).resolve().parent.parent / "model"


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    grid = {"n_estimators": [100, 300], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 3]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=-1), grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X_tr, y_tr)

    model = gs.best_estimator_
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    print(f"best params={gs.best_params_}  cv_auc={gs.best_score_:.4f}  test_auc={auc:.4f}")

    OUT.mkdir(exist_ok=True)
    joblib.dump(model, OUT / "champion_model.joblib", compress=3)
    meta = {
        "model_type": "RandomForestClassifier",
        "feature_names": list(X.columns),
        "classes": {"0": "malignant", "1": "benign"},
        "best_params": gs.best_params_,
        "cv_roc_auc": round(gs.best_score_, 4),
        "test_roc_auc": round(auc, 4),
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

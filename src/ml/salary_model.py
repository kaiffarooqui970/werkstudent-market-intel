"""
Salary regression model.

Target: log(salary_min_annualized_eur)  (log-transform to reduce right-skew)
Features:
  - Multi-hot skill vector (from bridge_job_skill)
  - City (one-hot, top-N cities + 'Other')
  - Role family (one-hot)
  - Seniority (one-hot)
  - Remote flag
  - Language gate (one-hot)

Model: XGBRegressor with time-series-aware CV (GroupKFold by snapshot_week) to
avoid the leakage of the same posting reappearing in multiple snapshots.

Interpretability: SHAP TreeExplainer. Global feature importance and per-prediction
force plots are exposed to the dashboard.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, RepeatedKFold
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

TOP_N_CITIES = 12
MIN_TRAINING_ROWS = 30  # absolute floor, regardless of feature count
MIN_SKILL_FREQ = 8  # skill must appear in at least this many postings to get its own feature
TOP_K_SKILLS = 20  # hard cap on skill features, on top of the frequency filter
MIN_GROUPS_FOR_GROUP_KFOLD = 3  # below this, grouped CV splits are too degenerate to trust
MIN_ROWS_PER_FEATURE = 5  # training floor scales with feature count, not a fixed constant

XGB_PARAMS: dict[str, Any] = dict(
    n_estimators=150,
    max_depth=3,
    min_child_weight=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42,
)


@dataclass
class ModelBundle:
    model: XGBRegressor
    feature_names: list[str]
    top_cities: list[str]
    all_skills: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    shap_global: dict[str, float] = field(default_factory=dict)

    def featurize_one(
        self,
        skills: list[str],
        city: str,
        role_family: str,
        seniority: str,
        remote: bool,
        language_gate: str,
    ) -> np.ndarray:
        row = pd.DataFrame([{
            "skills": skills,
            "city": city if city in self.top_cities else "Other",
            "role_family": role_family,
            "seniority": seniority,
            "remote": bool(remote),
            "language_gate": language_gate,
        }])
        X = _build_feature_matrix(row, self.top_cities, self.all_skills)
        # ensure column order matches training
        X = X.reindex(columns=self.feature_names, fill_value=0)
        return X.values


# Training runs as `python -m src.ml.salary_model`, which executes this file as
# __main__ and would otherwise pickle ModelBundle under "__main__" — unresolvable
# when unpickled from a different entry point (e.g. the Streamlit dashboard).
# Alias this module under its real dotted path so both the class's __module__
# and its lookup in sys.modules agree, regardless of how the file was invoked.
import sys as _sys

_sys.modules.setdefault("src.ml.salary_model", _sys.modules[__name__])
ModelBundle.__module__ = "src.ml.salary_model"


def _load_training_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT
            f.slug,
            f.city,
            f.role_family,
            f.seniority,
            f.remote,
            f.language_gate,
            f.salary_min_annualized_eur AS salary,
            f.snapshot_date,
            f.skills_json
        FROM fct_jobs f
        WHERE f.salary_min_annualized_eur IS NOT NULL
          AND f.salary_min_annualized_eur BETWEEN 10000 AND 250000
        """
    ).fetch_df()
    df["skills"] = df["skills_json"].fillna("[]").map(json.loads)
    return df.drop(columns=["skills_json"])


def _build_feature_matrix(
    df: pd.DataFrame, top_cities: list[str], all_skills: list[str]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    # skills multi-hot
    skill_mat = pd.DataFrame(0, index=df.index, columns=[f"skill__{s}" for s in all_skills])
    for i, sks in enumerate(df["skills"]):
        for s in sks:
            col = f"skill__{s}"
            if col in skill_mat.columns:
                skill_mat.iat[i, skill_mat.columns.get_loc(col)] = 1
    frames.append(skill_mat)

    # categoricals
    city_series = df["city"].where(df["city"].isin(top_cities), "Other")
    frames.append(pd.get_dummies(city_series, prefix="city"))
    frames.append(pd.get_dummies(df["role_family"].fillna("Other"), prefix="role"))
    frames.append(pd.get_dummies(df["seniority"].fillna("unknown"), prefix="sen"))
    frames.append(pd.get_dummies(df["language_gate"].fillna("unclear"), prefix="lang"))
    frames.append(df[["remote"]].astype(int).rename(columns={"remote": "remote"}))

    X = pd.concat(frames, axis=1)
    X.columns = [str(c) for c in X.columns]
    return X


def _cv_metrics(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    n_groups = len(np.unique(groups))

    if n_groups >= MIN_GROUPS_FOR_GROUP_KFOLD:
        n_splits = min(5, n_groups)
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups)
        fold_label = "n_folds"
        fold_count = n_splits
    else:
        # Too few distinct snapshot weeks for grouped CV to mean anything (often
        # just 1-2 groups on effectively single-snapshot data). Repeated shuffled
        # KFold gives a stable estimate instead of one arbitrary, unshuffled split.
        n_splits = 5
        n_repeats = 5
        splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        split_iter = splitter.split(X, y)
        fold_label = "n_repeats"
        fold_count = n_repeats

    maes, r2s = [], []
    n_splits_run = 0
    for train_idx, test_idx in split_iter:
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])
        maes.append(mean_absolute_error(y[test_idx], pred))
        if len(pred) > 1:
            r2s.append(r2_score(y[test_idx], pred))
        n_splits_run += 1

    return {
        "mae_log": float(np.mean(maes)),
        "r2": float(np.mean(r2s)) if r2s else float("nan"),
        fold_label: fold_count,
        "n_splits": n_splits_run,
    }


def train(db_path: str, out_path: str) -> ModelBundle | None:
    con = duckdb.connect(db_path)
    df = _load_training_frame(con)
    con.close()

    # top cities by frequency
    top_cities = df["city"].value_counts().head(TOP_N_CITIES).index.tolist()

    # skill vocabulary = anything appearing in >= MIN_SKILL_FREQ postings,
    # capped to the TOP_K_SKILLS most frequent so feature count stays well
    # below row count even on small datasets
    from collections import Counter
    ctr: Counter[str] = Counter()
    for sks in df["skills"]:
        ctr.update(sks)
    all_skills = sorted(
        [s for s, _ in ctr.most_common(TOP_K_SKILLS) if ctr[s] >= MIN_SKILL_FREQ]
    )
    if not all_skills:
        all_skills = sorted({s for sks in df["skills"] for s in sks})

    X_df = _build_feature_matrix(df, top_cities, all_skills)
    n_features = X_df.shape[1]
    min_training_rows = max(MIN_TRAINING_ROWS, n_features * MIN_ROWS_PER_FEATURE)

    if len(df) < min_training_rows:
        logger.warning(
            "only %d rows with salary against %d features; need >= %d "
            "(%d rows/feature). Skipping model training.",
            len(df), n_features, min_training_rows, MIN_ROWS_PER_FEATURE,
        )
        return None

    y = np.log(df["salary"].values.astype(float))

    # time-series-ish grouping: bucket by snapshot week to keep repostings together
    groups = pd.to_datetime(df["snapshot_date"]).dt.strftime("%Y-%U").values

    metrics = _cv_metrics(X_df.values, y, groups)

    # refit on all data for the persisted model
    final = XGBRegressor(**XGB_PARAMS)
    final.fit(X_df.values, y)

    # SHAP global importance (mean |shap|)
    explainer = shap.TreeExplainer(final)
    shap_vals = explainer.shap_values(X_df.values)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    shap_global = dict(
        sorted(zip(X_df.columns.tolist(), mean_abs.tolist()), key=lambda kv: -kv[1])[:25]
    )

    bundle = ModelBundle(
        model=final,
        feature_names=X_df.columns.tolist(),
        top_cities=top_cities,
        all_skills=all_skills,
        metrics=metrics,
        shap_global=shap_global,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    logger.info("model saved to %s | metrics=%s", out_path, metrics)
    return bundle


def load(path: str) -> ModelBundle:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_eur(bundle: ModelBundle, **inputs: Any) -> float:
    X = bundle.featurize_one(**inputs)
    log_pred = bundle.model.predict(X)[0]
    return float(np.exp(log_pred))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/warehouse.duckdb")
    p.add_argument("--out", default="data/model.pkl")
    args = p.parse_args()
    train(args.db, args.out)


if __name__ == "__main__":
    main()

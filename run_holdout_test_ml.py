"""
Single-Pass Stage 2 ML Benchmark: 80/20 Stratified Holdout Evaluation.
Trains each model once, evaluates on the 20% holdout test set,
and passes the fitted models directly to save_holdout_models() 
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.save_holdout_checkpoints import save_holdout_models

FEATURE_COLS = [
    "early_suspicion_score",
    "newman_weight",
    "artifact_degree",
    "topological_gravity_bm25",
    "topological_risk_nist",
    "causal_depth",
    "payload_char_len",
    "is_stigmergic_reader",
]


def resolve_dataset_path(filename: str) -> Path:
    candidates = [
        Path("dataset") / filename,
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Could not locate {filename} in working or dataset directories.")


def load_and_preprocess_data(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Ingests CSV, extracts X and y, applies 80/20 stratified split, and fits the scaler."""
    df = pd.read_csv(csv_path)
    target_col = "is_malicious" if "is_malicious" in df.columns else "y"
    if target_col not in df.columns:
        raise KeyError(f"Target column not found. Available columns: {list(df.columns)}")

    X = df[FEATURE_COLS].copy()
    y = df[target_col].astype(int).values

    # Stratified 80/20 Holdout Partition
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Feature scaling fitted strictly on X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train.values, X_test.values, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """Fits model once, benchmarks inference latency, and computes metrics."""
    # Single fit pass
    model.fit(X_train, y_train)

    # Latency benchmark
    t_start = time.perf_counter()
    y_pred = model.predict(X_test)
    t_elapsed = time.perf_counter() - t_start
    latency_per_sample_us = (t_elapsed / len(X_test)) * 1_000_000.0

    # Probability predictions for PR-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred

    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    pr_auc = average_precision_score(y_test, y_proba)

    return {
        "model_name": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "latency_us": latency_per_sample_us,
    }


def run_mode_benchmark(csv_filename: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], StandardScaler]:
    """Runs evaluation on a dataset mode and retains the fitted models in memory."""
    csv_path = resolve_dataset_path(csv_filename)
    X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te, scaler = load_and_preprocess_data(csv_path)

    scale_pos = float(np.sum(y_tr == 0) / np.sum(y_tr == 1))

    # Base estimators
    rf_base = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
    xgb_base = XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42, n_jobs=-1)

    models_config = [
        ("Logistic Regression (Baseline)", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42), True),
        ("ANN (Multi-Layer Perceptron)", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=200, early_stopping=True, random_state=42), True),
        ("Random Forest (Balanced)", rf_base, False),
        ("XGBoost (Scale Pos Weight)", xgb_base, False),
        ("TraceSentry Ensemble (RF + XGB)", VotingClassifier(estimators=[("rf", rf_base), ("xgb", xgb_base)], voting="soft", weights=[0.5, 0.5], n_jobs=-1), False),
    ]

    results = []
    fitted_models: Dict[str, Any] = {}

    for name, model, use_scaled in models_config:
        X_train_in = X_tr_s if use_scaled else X_tr
        X_test_in = X_te_s if use_scaled else X_te

        res = evaluate_model(model, X_train_in, y_tr, X_test_in, y_te, name)
        results.append(res)
        fitted_models[name] = model  # Retain the fitted model instance

    return results, fitted_models, scaler


def print_comparison_table(egress_results: List[Dict[str, Any]], dpi_results: List[Dict[str, Any]]) -> None:
    """Formats and prints the exact comparison table."""
    print("=" * 104)
    print("STAGE 2 HOLDOUT TEST EVALUATION TABLE (20% Pristine Test Set, N ~ 10,149 / 12,057)")
    print("=" * 104)
    print(f"{'Model Architecture':<34} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'PR-AUC':<8} | {'Latency / Sample':<16}")
    print("-" * 104)

    print("[EGRESS_ONLY MODE]")
    for r in egress_results:
        latency_str = f"{r['latency_us']:.2f} µs" if r['latency_us'] < 1000 else f"{r['latency_us'] / 1000.0:.3f} ms"
        print(f"  {r['model_name']:<32} | {r['precision']:<9.4f} | {r['recall']:<8.4f} | {r['f1']:<8.4f} | {r['pr_auc']:<8.4f} | {latency_str:<16}")

    print("-" * 104)
    print("[FULL_DPI MODE]")
    for r in dpi_results:
        latency_str = f"{r['latency_us']:.2f} µs" if r['latency_us'] < 1000 else f"{r['latency_us'] / 1000.0:.3f} ms"
        print(f"  {r['model_name']:<32} | {r['precision']:<9.4f} | {r['recall']:<8.4f} | {r['f1']:<8.4f} | {r['pr_auc']:<8.4f} | {latency_str:<16}")
    print("=" * 104)


if __name__ == "__main__":
    # Evaluate and Train
    print("[*] Training and Evaluating Stage 2 Classifiers on EGRESS_ONLY dataset...")
    egress_res, egress_models, egress_scaler = run_mode_benchmark("stage2_features_egress_only.csv")

    print("[*] Training and Evaluating Stage 2 Classifiers on FULL_DPI dataset...")
    dpi_res, dpi_models, dpi_scaler = run_mode_benchmark("stage2_features_full_dpi.csv")

    # Print evaluation metrics
    print_comparison_table(egress_res, dpi_res)

    # Save the already-fitted models directly to disk
    output_dir = Path("checkpoints/holdout")
    print("\n" + "=" * 75)
    print("SERIALIZING TRAINED HOLDOUT CHECKPOINTS (Zero Re-Training)")
    print("=" * 75)

    print("\n[*] Saving EGRESS_ONLY Models...")
    save_holdout_models(egress_models, egress_scaler, "egress_only", output_dir=output_dir)

    print("\n[*] Saving FULL_DPI Models...")
    save_holdout_models(dpi_models, dpi_scaler, "full_dpi", output_dir=output_dir)

    print("\n" + "=" * 75)
    print(f"[✔] All Holdout Checkpoints saved successfully to: {output_dir.resolve()}")
    print("=" * 75)

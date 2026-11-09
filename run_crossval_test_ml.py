"""
Single-Pass Stage 2 ML Benchmark: 5-Fold Stratified Cross-Validation.
Trains each model once across 5 folds on the 80% train partition,
computes Mean ± Std metrics, and directly serializes all 50 models + 10 scalers
via save_crossval_models() with zero duplicate training.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.save_crossval_checkpoints import save_crossval_models

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
        Path("/kaggle/working/dataset") / filename,
        Path("/kaggle/working") / filename,
        Path("dataset") / filename,
        Path(filename),
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Could not locate {filename} in working or dataset directories.")


def load_train_partition(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Ingests CSV and extracts strictly the 80% Training partition."""
    df = pd.read_csv(csv_path)
    target_col = "is_malicious" if "is_malicious" in df.columns else "y"
    if target_col not in df.columns:
        raise KeyError(f"Target column not found. Available columns: {list(df.columns)}")

    X = df[FEATURE_COLS].copy().values
    y = df[target_col].astype(int).values

    # Identical 80/20 Stratified Partition as Block 1
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    return X_train, y_train


def run_5fold_cross_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    mode_label: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Executes 5-fold cross-validation in a single pass.
    Returns:
        aggregated_results: List of metric dicts (Mean ± Std).
        fold_artifacts: List of dicts containing the fitted models and fold scalers.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_names = [
        "Logistic Regression (Baseline)",
        "ANN (Multi-Layer Perceptron)",
        "Random Forest (Balanced)",
        "XGBoost (Scale Pos Weight)",
        "TraceSentry Ensemble (RF + XGB)",
    ]

    f1_records: Dict[str, List[float]] = {name: [] for name in model_names}
    prec_records: Dict[str, List[float]] = {name: [] for name in model_names}
    rec_records: Dict[str, List[float]] = {name: [] for name in model_names}

    fold_artifacts: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        print(f"  [{mode_label}] Training & Evaluating Fold {fold_idx}/5...")
        X_tr_f, X_val_f = X_train[train_idx], X_train[val_idx]
        y_tr_f, y_val_f = y_train[train_idx], y_train[val_idx]

        # 1. Fold-isolated feature scaling
        scaler_fold = StandardScaler()
        X_tr_f_s = scaler_fold.fit_transform(X_tr_f)
        X_val_f_s = scaler_fold.transform(X_val_f)

        # Dynamic class weight for XGBoost inside this fold
        scale_pos_fold = float(np.sum(y_tr_f == 0) / np.sum(y_tr_f == 1))

        # Base Estimators
        rf_fold = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42
        )
        xgb_fold = XGBClassifier(
            n_estimators=100,
            scale_pos_weight=scale_pos_fold,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        fold_models = [
            (
                "Logistic Regression (Baseline)",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
                X_tr_f_s,
                X_val_f_s,
            ),
            (
                "ANN (Multi-Layer Perceptron)",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=200,
                    early_stopping=True,
                    random_state=42,
                ),
                X_tr_f_s,
                X_val_f_s,
            ),
            ("Random Forest (Balanced)", rf_fold, X_tr_f, X_val_f),
            ("XGBoost (Scale Pos Weight)", xgb_fold, X_tr_f, X_val_f),
            (
                "TraceSentry Ensemble (RF + XGB)",
                VotingClassifier(
                    estimators=[("rf", rf_fold), ("xgb", xgb_fold)],
                    voting="soft",
                    weights=[0.5, 0.5],
                    n_jobs=-1,
                ),
                X_tr_f,
                X_val_f,
            ),
        ]

        fold_fitted_models: Dict[str, Any] = {}

        # Train each model once and evaluate on validation fold
        for name, model, x_train_in, x_val_in in fold_models:
            model.fit(x_train_in, y_tr_f)
            y_pred = model.predict(x_val_in)

            f1_records[name].append(f1_score(y_val_f, y_pred, pos_label=1, zero_division=0))
            prec_records[name].append(precision_score(y_val_f, y_pred, pos_label=1, zero_division=0))
            rec_records[name].append(recall_score(y_val_f, y_pred, pos_label=1, zero_division=0))

            fold_fitted_models[name] = model

        # Preserve the fold's trained artifacts in memory
        fold_artifacts.append({
            "fold_idx": fold_idx,
            "scaler": scaler_fold,
            "models": fold_fitted_models
        })

    # Aggregate Mean and Standard Deviation across 5 folds
    aggregated_results = []
    for name in model_names:
        aggregated_results.append({
            "model_name": name,
            "f1_mean": float(np.mean(f1_records[name])),
            "f1_std": float(np.std(f1_records[name])),
            "prec_mean": float(np.mean(prec_records[name])),
            "rec_mean": float(np.mean(rec_records[name])),
        })

    return aggregated_results, fold_artifacts


def print_crossval_table(
    egress_cv: List[Dict[str, Any]],
    dpi_cv: List[Dict[str, Any]]
) -> None:
    """Formats and prints the 5-fold cross-validation stability report."""
    print("=" * 104)
    print("5-FOLD STRATIFIED CROSS-VALIDATION STABILITY REPORT (Executed on 80% Training Partition)")
    print("=" * 104)
    print(f"{'Model Architecture':<34} | {'CV F1-Score (Mean ± Std)':<24} | {'CV Precision (Mean)':<19} | {'CV Recall (Mean)':<16}")
    print("-" * 104)

    print("[EGRESS_ONLY MODE]")
    for r in egress_cv:
        f1_str = f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
        print(f"  {r['model_name']:<32} | {f1_str:<24} | {r['prec_mean']:<19.4f} | {r['rec_mean']:<16.4f}")

    print("-" * 104)
    print("[FULL_DPI MODE]")
    for r in dpi_cv:
        f1_str = f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
        print(f"  {r['model_name']:<32} | {f1_str:<24} | {r['prec_mean']:<19.4f} | {r['rec_mean']:<16.4f}")
    print("=" * 104)


if __name__ == "__main__":
    # EGRESS_ONLY Mode (Train, Evaluate, and Collect in 1 Pass)
    print("[*] Loading 80% Training Partition for EGRESS_ONLY dataset...")
    egress_path = resolve_dataset_path("stage2_features_egress_only.csv")
    X_train_eg, y_train_eg = load_train_partition(egress_path)
    print(f"    - Training Shape: {X_train_eg.shape} | Positive Samples: {np.sum(y_train_eg == 1):,}")

    print("[*] Running 5-Fold Stratified Cross-Validation on EGRESS_ONLY...")
    egress_cv_results, egress_fold_artifacts = run_5fold_cross_validation(X_train_eg, y_train_eg, "EGRESS_ONLY")

    # FULL_DPI Mode (Train, Evaluate, and Collect in 1 Pass)
    print("\n[*] Loading 80% Training Partition for FULL_DPI dataset...")
    dpi_path = resolve_dataset_path("stage2_features_full_dpi.csv")
    X_train_dp, y_train_dp = load_train_partition(dpi_path)
    print(f"    - Training Shape: {X_train_dp.shape} | Positive Samples: {np.sum(y_train_dp == 1):,}")

    print("[*] Running 5-Fold Stratified Cross-Validation on FULL_DPI...")
    dpi_cv_results, dpi_fold_artifacts = run_5fold_cross_validation(X_train_dp, y_train_dp, "FULL_DPI")

    # Print the CV Stability Table to Terminal
    print()
    print_crossval_table(egress_cv_results, dpi_cv_results)

    # Serialize all 50 models + 10 scalers directly from memory
    output_dir = Path("checkpoints/crossval")
    print("\n" + "=" * 75)
    print("SERIALIZING 5-FOLD CROSS-VALIDATION CHECKPOINTS (Zero Re-Training)")
    print("=" * 75)

    print("\n[*] Saving EGRESS_ONLY 5-Fold Checkpoints...")
    save_crossval_models(egress_fold_artifacts, "egress_only", output_dir=output_dir)

    print("\n[*] Saving FULL_DPI 5-Fold Checkpoints...")
    save_crossval_models(dpi_fold_artifacts, "full_dpi", output_dir=output_dir)

    print("\n" + "=" * 75)
    print(f"[✔] All 5-Fold Checkpoints saved successfully to: {output_dir.resolve()}")
    print(f"    Total files exported: {len(list(output_dir.glob('*.joblib')))}")
    print("=" * 75)

"""
Pure I/O serialization module for cross-validation checkpoints.
Accepts already-fitted fold models and fold-specific StandardScalers from the CV runner
and serializes them to disk using joblib. Contains zero training or data loading.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import joblib
from sklearn.preprocessing import StandardScaler

# Mapping display names to standardized filename slugs
SLUG_MAP = {
    "Logistic Regression (Baseline)": "logistic_regression",
    "ANN (Multi-Layer Perceptron)": "ann_mlp",
    "Random Forest (Balanced)": "random_forest",
    "XGBoost (Scale Pos Weight)": "xgboost",
    "TraceSentry Ensemble (RF + XGB)": "tracesentry_ensemble",
}


def save_crossval_models(
    fold_artifacts: List[Dict[str, Any]],
    mode_prefix: str,
    output_dir: str | Path = "checkpoints/crossval"
) -> None:
    """
    Serializes already-trained fold models and their paired fold StandardScalers.

    Args:
        fold_artifacts: List of dicts, each containing:
            {
                "fold_idx": int,
                "scaler": StandardScaler,
                "models": Dict[str, Any]
            }
        mode_prefix: 'egress_only' or 'full_dpi'.
        output_dir: Destination folder path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for artifact in fold_artifacts:
        fold_idx = artifact["fold_idx"]
        scaler = artifact["scaler"]
        models_dict = artifact["models"]

        # Save fold-specific Scaler
        scaler_file = out_path / f"{mode_prefix}_fold{fold_idx}_scaler.joblib"
        joblib.dump(scaler, scaler_file)
        print(f"[+] Saved Scaler: {scaler_file.name}")

        # 2. Save each fold model
        for name, model in models_dict.items():
            slug = SLUG_MAP.get(name, name.lower().replace(" ", "_").replace("-", "_"))
            model_file = out_path / f"{mode_prefix}_fold{fold_idx}_{slug}.joblib"
            joblib.dump(model, model_file)
            print(f"[+] Saved Model:  {model_file.name}")

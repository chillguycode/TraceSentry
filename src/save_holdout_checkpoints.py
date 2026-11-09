"""
Pure I/O serialization module.
Accepts already-trained models and fitted StandardScaler from the evaluation runner
and serializes them to disk using joblib. Contains zero training or data loading.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
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


def save_holdout_models(
    trained_models: Dict[str, Any],
    scaler: StandardScaler,
    mode_prefix: str,
    output_dir: str | Path = "checkpoints/holdout"
) -> None:
    """
    Serializes already-trained model instances and their paired StandardScaler.
    
    Args:
        trained_models: Dictionary mapping model names to fitted model objects.
        scaler: The fitted StandardScaler from the training split.
        mode_prefix: 'egress_only' or 'full_dpi'.
        output_dir: Destination folder path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save the fitted Scaler
    scaler_file = out_path / f"{mode_prefix}_scaler.joblib"
    joblib.dump(scaler, scaler_file)
    print(f"[+] Saved Scaler: {scaler_file.name}")

    # 2. Save each trained model directly from memory
    for name, model in trained_models.items():
        slug = SLUG_MAP.get(name, name.lower().replace(" ", "_").replace("-", "_"))
        model_file = out_path / f"{mode_prefix}_{slug}.joblib"
        joblib.dump(model, model_file)
        print(f"[+] Saved Model:  {model_file.name}")

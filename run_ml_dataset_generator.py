"""
src/dataset_generator.py
Stage 2 Feature Extraction and Tabular Dataset Generator.
Extracts 8 engineered topological and causal features from escalated events:
1. early_suspicion_score (Float)
2. newman_weight (Float)
3. artifact_degree (Integer)
4. topological_gravity_bm25 (Float)
5. topological_risk_nist (Float)
6. causal_depth (Integer)
7. payload_char_len (Integer)
8. is_stigmergic_reader (Integer, 0 or 1)
Exports clean, validated CSV files for Classical ML training.
"""

from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.early_checker import GatewayEarlyChecker, EarlyCheckNotice
from src.graph_builder import DiamondFeatureExtractor
from src.schema import SwarmTracesRawRecord


# Ground Truth Attack Set Resolution
def get_confirmed_attack_ids(dataset_path: str | Path) -> Set[str]:
    """
    Returns the confirmed attack payload IDs from notebook globals if available,
    or extracts the ground-truth attack cluster from dataset provenance markers.
    """
    if "known_attack_ids" in globals() and isinstance(globals()["known_attack_ids"], set):
        print(f"[+] Using existing 'known_attack_ids' from session ({len(globals()['known_attack_ids']):,} IDs).")
        return globals()["known_attack_ids"]

    print("[*] Extracting confirmed attack payload IDs from dataset markers...")
    attack_ids = set()
    path = Path(dataset_path)

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            row_id = item.get("id", item.get("row_id", f"R_{idx}"))
            text = item.get("text", "")

            # Confirmed compromise assets established during graph provenance attribution
            if (
                "[CREDENTIAL 1]" in text
                or "[REDACTED:source_identifier:000001]" in text
                or "[REDACTED:source_identifier:000002]" in text
                or "[REDACTED:source_identifier:000003]" in text
                or "G23619" in text
                or "D64LINK" in text
                or "LOOT" in text
                or ";id;" in text
                or "uname -a" in text
                or "/proc/1/status" in text
            ):
                attack_ids.add(row_id)

    print(f"[+] Confirmed Attack Ground Truth: {len(attack_ids):,} IDs.")
    return attack_ids


# Global Asset Degree Pre-computation 
def compute_asset_degrees(dataset_path: str | Path) -> Dict[str, int]:
    """
    Pre-computes artifact degrees (d_k) across the entire corpus in a single pass.
    Guarantees O(1) feature lookups without storing 189k graph nodes in memory.
    """
    print("[*] Pre-computing global artifact degrees (d_k) from trace corpus...")
    degree_counter: Dict[str, int] = {}
    path = Path(dataset_path)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item.get("text", "")
            if not text:
                continue

            infra_items = DiamondFeatureExtractor.extract_infrastructure(text)
            seen_in_event = set()
            for subtype, val in infra_items:
                artifact_key = f"{subtype}:{val}"
                if artifact_key not in seen_in_event:
                    seen_in_event.add(artifact_key)
                    degree_counter[artifact_key] = degree_counter.get(artifact_key, 0) + 1

    print(f"[+] Pre-computed degrees for {len(degree_counter):,} distinct artifacts.")
    return degree_counter


# Feature Extraction Engine
def extract_stage2_features(
    record: SwarmTracesRawRecord,
    notice: EarlyCheckNotice,
    asset_degrees: Dict[str, int],
    is_malicious: int,
    mode: str
) -> Dict[str, Any]:
    """
    Computes the 8 numeric Stage 2 features for an escalated event.
    Enforces the strict ablation boundary for EGRESS_ONLY mode.
    """
    text = record.text or ""
    infra_items = DiamondFeatureExtractor.extract_infrastructure(text)

    # Ablation Boundary: In EGRESS_ONLY mode, strictly filter out body checksums
    if mode == "EGRESS_ONLY":
        candidate_assets = [(stype, val) for stype, val in infra_items if stype != "checksum"]
    else:
        candidate_assets = infra_items

    # Feature - early_suspicion_score (Float [0.20, 1.0])
    early_score = round(notice.suspicion_score, 4)

    # Evaluate asset metrics across touched artifacts to find the primary asset
    best_degree = 0
    best_newman = 0.0
    best_gravity = 0.0
    best_risk = 0.0
    has_credentials = False
    has_checksums = False

    alpha_weights = {
        "credential": 5.0,
        "checksum": 4.5,
        "board_or_c2": 4.5,
        "task_id": 3.0,
        "destination": 1.0
    }

    for stype, val in candidate_assets:
        if stype == "credential":
            has_credentials = True
        elif stype == "checksum":
            has_checksums = True

        artifact_key = f"{stype}:{val}"
        d_k = asset_degrees.get(artifact_key, 1)
        alpha = alpha_weights.get(stype, 1.0)

        # Feature -  newman_weight 
        w_newman = (1.0 / (d_k - 1)) if d_k > 1 else 0.0

        # Feature - topological_gravity_bm25 
        gravity_bm25 = alpha * (d_k / (d_k + 20.0))

        # Feature - topological_risk_nist
        risk_nist = alpha * math.log10(1.0 + d_k)

        # Retain maximum gravity among multiple touched assets
        if gravity_bm25 > best_gravity:
            best_gravity = gravity_bm25
            best_risk = risk_nist
            best_newman = w_newman
            best_degree = d_k

    # Feature - causal_depth
    causal_depth = int(record.depth) if record.depth is not None else 0

    # Feature - payload_char_len
    payload_char_len = len(text)

    # Feature - is_stigmergic_reader
    # Event inherited assets via shared board/storage with no direct lineage parent
    has_lineage_parents = len(record.source_rows) > 0
    is_stigmergic = 1 if (not has_lineage_parents and (has_credentials or has_checksums)) else 0

    return {
        "row_id": record.id,
        "early_suspicion_score": early_score,
        "newman_weight": round(best_newman, 6),
        "artifact_degree": int(best_degree),
        "topological_gravity_bm25": round(best_gravity, 6),
        "topological_risk_nist": round(best_risk, 6),
        "causal_depth": causal_depth,
        "payload_char_len": payload_char_len,
        "is_stigmergic_reader": is_stigmergic,
        "is_malicious": int(is_malicious)
    }


# Streaming Export Pipeline
def export_stage2_dataset(
    dataset_path: str | Path,
    output_csv_path: str | Path,
    mode: str,
    asset_degrees: Dict[str, int],
    attack_ids: Set[str],
    limit: Optional[int] = None
) -> Tuple[int, int, int]:
    """
    Streams the dataset through GatewayEarlyChecker, extracts the 8 features
    for escalated rows, validates data integrity, and writes the CSV file.
    """
    path = Path(dataset_path)
    out_file = Path(output_csv_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    checker = GatewayEarlyChecker(mode=mode)
    fieldnames = [
        "row_id",
        "early_suspicion_score",
        "newman_weight",
        "artifact_degree",
        "topological_gravity_bm25",
        "topological_risk_nist",
        "causal_depth",
        "payload_char_len",
        "is_stigmergic_reader",
        "is_malicious"
    ]

    total_exported = 0
    positive_count = 0
    negative_count = 0
    sample_rows: List[Dict[str, Any]] = []

    print(f"\n[*] Generating Stage 2 Dataset: [{mode}] -> {out_file.name} ...")

    with open(path, "r", encoding="utf-8") as f_in, open(out_file, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx, line in enumerate(f_in):
            if limit and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue

            record_dict = json.loads(line)
            record = SwarmTracesRawRecord(**record_dict)
            seq_index = DiamondFeatureExtractor.extract_seq_index(record.id, fallback_step=idx)

            # Stage 1 Screening
            notice = checker.evaluate_event(record, seq_index=seq_index)

            # Stage 2 Extraction: Only process escalated candidate events (Score >= 0.20)
            if notice.escalate:
                is_malicious = 1 if record.id in attack_ids else 0
                feature_row = extract_stage2_features(
                    record=record,
                    notice=notice,
                    asset_degrees=asset_degrees,
                    is_malicious=is_malicious,
                    mode=mode
                )

                # Integrity Invariant: Assert zero NaN or Infinite values
                for col in fieldnames[1:]:
                    val = feature_row[col]
                    assert not (math.isnan(val) or math.isinf(val)), f"Invalid numeric value in {col}: {val}"

                writer.writerow(feature_row)
                total_exported += 1
                if is_malicious == 1:
                    positive_count += 1
                else:
                    negative_count += 1

                if len(sample_rows) < 3:
                    sample_rows.append(feature_row)

    # Print Verification Summary Table
    pos_pct = (positive_count / total_exported * 100.0) if total_exported > 0 else 0.0
    neg_pct = (negative_count / total_exported * 100.0) if total_exported > 0 else 0.0

    print("=" * 80)
    print(f"VERIFICATION SUMMARY: {out_file.name}")
    print("=" * 80)
    print(f"  - Total Escalated Rows Exported: {total_exported:,}")
    print(f"  - Positive Samples (y=1):        {positive_count:,} ({pos_pct:.2f}%)")
    print(f"  - Negative Samples (y=0):        {negative_count:,} ({neg_pct:.2f}%)")
    print("  - Numerical Integrity Check:     PASS (0 NaN, 0 Null, 0 Inf across all columns)")
    print("\n  Sample Feature Rows:")
    for s_idx, sample in enumerate(sample_rows, 1):
        print(f"   [{s_idx}] ID={sample['row_id']} | Score={sample['early_suspicion_score']:.2f} | "
              f"Newman={sample['newman_weight']:.4f} | Deg={sample['artifact_degree']} | "
              f"Grav={sample['topological_gravity_bm25']:.3f} | NIST={sample['topological_risk_nist']:.3f} | "
              f"Depth={sample['causal_depth']} | Stigm={sample['is_stigmergic_reader']} | y={sample['is_malicious']}")
    print("=" * 80)

    return total_exported, positive_count, negative_count


if __name__ == "__main__":
    DATASET_PATH = "dataset/redacted.jsonl"

    OUTPUT_DIR = Path("dataset")

    # Resolve confirmed attack IDs
    confirmed_attack_ids = get_confirmed_attack_ids(DATASET_PATH)

    # Pre-compute global artifact degrees
    global_asset_degrees = compute_asset_degrees(DATASET_PATH)

    # Export Dataset 1: EGRESS_ONLY
    egress_csv = OUTPUT_DIR / "stage2_features_egress_only.csv"
    export_stage2_dataset(
        dataset_path=DATASET_PATH,
        output_csv_path=egress_csv,
        mode="EGRESS_ONLY",
        asset_degrees=global_asset_degrees,
        attack_ids=confirmed_attack_ids,
        limit=None  # Full dataset run
    )

    # Export Dataset 2: FULL_DPI
    dpi_csv = OUTPUT_DIR / "stage2_features_full_dpi.csv"
    export_stage2_dataset(
        dataset_path=DATASET_PATH,
        output_csv_path=dpi_csv,
        mode="FULL_DPI",
        asset_degrees=global_asset_degrees,
        attack_ids=confirmed_attack_ids,
        limit=None  # Full dataset run
    )

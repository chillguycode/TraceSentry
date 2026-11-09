"""
Command-Line Interface (CLI) Benchmark Runner for GatewayEarlyChecker.
Evaluates the 5 core telemetry variables:
1. Time Accumulator (Throughput & Latency)
2. Volume Counters (Stage 2 Reduction Ratio)
3. First-Alert Sequence Marker (Early-Warning Lead Time)
4. Flagged ID Set (Attribution Volume)
5. Rule Attribution Breakdown (Ablation Distribution)

Default Behavior (Zero Arguments):
    Runs BOTH ablation modes (FULL_DPI and EGRESS_ONLY) across the ENTIRE dataset.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Set
from pydantic import BaseModel

# Defensive imports supporting both local package structure and root directory execution
from src.early_checker import GatewayEarlyChecker, EarlyCheckNotice
from src.graph_builder import DiamondFeatureExtractor
from src.schema import SwarmTracesRawRecord


class BenchmarkTelemetry(BaseModel):
    """
    Structured container for the 5 benchmark telemetry criteria.
    """
    mode: str
    total_elapsed_seconds: float
    avg_latency_ms_per_event: float
    throughput_events_per_sec: float
    total_ingested: int
    escalated_to_stage_2: int
    dropped_on_fast_path: int
    stage_2_reduction_ratio_pct: float
    first_alert_seq_index: Optional[int]
    first_alert_row_id: Optional[str]
    early_warning_lead_time_pct: float
    flagged_ids_count: int
    rule_attribution_counts: Dict[str, int]
    tainted_assets_count: int


def run_earlychecker_benchmark(
    dataset_path: str | Path,
    mode: str = "FULL_DPI",
    limit: Optional[int] = None
) -> BenchmarkTelemetry:
    """
    Streams events through GatewayEarlyChecker and computes the 5 telemetry metrics.
    """
    path = Path(dataset_path)
    if not path.is_file():
        print(f"[!] Error: Dataset file not found at: {path.resolve()}")
        sys.exit(1)

    checker = GatewayEarlyChecker(mode=mode)

    # Time Accumulator Initialization
    t_start = time.perf_counter()

    # 2. Volume Counters Initialization
    total_ingested = 0
    escalated_to_stage_2 = 0
    dropped_on_fast_path = 0

    # First-Alert Sequence Marker Initialization
    first_alert_seq_index: Optional[int] = None
    first_alert_row_id: Optional[str] = None

    # Flagged ID Set Initialization
    flagged_ids: Set[str] = set()

    # Rule Attribution Breakdown Initialization
    rule_counter: Counter[str] = Counter()

    # Line-by-line streaming loop 
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue

            record_dict = json.loads(line)
            record = SwarmTracesRawRecord(**record_dict)
            seq_index = DiamondFeatureExtractor.extract_seq_index(record.id, fallback_step=idx)

            # Evaluate through the Early Checker
            notice: EarlyCheckNotice = checker.evaluate_event(record, seq_index=seq_index)
            total_ingested += 1

            if notice.escalate:
                escalated_to_stage_2 += 1
                flagged_ids.add(notice.row_id)

                if first_alert_seq_index is None:
                    first_alert_seq_index = seq_index
                    first_alert_row_id = notice.row_id

                for rule in notice.tripped_rules:
                    rule_counter[rule] += 1
            else:
                dropped_on_fast_path += 1

    t_elapsed = time.perf_counter() - t_start

    # Derived Telemetry Metrics
    latency_ms = (t_elapsed / total_ingested * 1000.0) if total_ingested > 0 else 0.0
    throughput = (total_ingested / t_elapsed) if t_elapsed > 0 else 0.0
    reduction_ratio = (dropped_on_fast_path / total_ingested * 100.0) if total_ingested > 0 else 0.0

    lead_time_pct = 0.0
    if first_alert_seq_index is not None and total_ingested > 0:
        lead_time_pct = (1.0 - (first_alert_seq_index / total_ingested)) * 100.0

    return BenchmarkTelemetry(
        mode=mode,
        total_elapsed_seconds=round(t_elapsed, 4),
        avg_latency_ms_per_event=round(latency_ms, 5),
        throughput_events_per_sec=round(throughput, 1),
        total_ingested=total_ingested,
        escalated_to_stage_2=escalated_to_stage_2,
        dropped_on_fast_path=dropped_on_fast_path,
        stage_2_reduction_ratio_pct=round(reduction_ratio, 2),
        first_alert_seq_index=first_alert_seq_index,
        first_alert_row_id=first_alert_row_id,
        early_warning_lead_time_pct=round(lead_time_pct, 4),
        flagged_ids_count=len(flagged_ids),
        rule_attribution_counts=dict(rule_counter.most_common()),
        tainted_assets_count=len(checker.taint_registry)
    )


def print_telemetry_report(t: BenchmarkTelemetry) -> None:
    """Formats and prints the 5 telemetry metrics into a clean paper-ready table."""
    print("=" * 80)
    print(f"EARLY CHECKER TELEMETRY REPORT: MODE [{t.mode}]")
    print("=" * 80)

    print("\n[1] TIME ACCUMULATOR (Throughput & Wire-Speed Latency)")
    print(f"    - Total Time Elapsed:        {t.total_elapsed_seconds:.4f} s")
    print(f"    - Avg Latency Per Event:     {t.avg_latency_ms_per_event:.5f} ms (< 0.05 ms SLA)")
    print(f"    - Ingestion Throughput:      {t.throughput_events_per_sec:,.1f} events/sec")

    print("\n[2] VOLUME COUNTERS (Stage 1 Filtering Efficiency)")
    print(f"    - Total Events Ingested:     {t.total_ingested:,}")
    print(f"    - Dropped on Fast-Path:      {t.dropped_on_fast_path:,} (Clean traffic filtered)")
    print(f"    - Escalated to Stage 2:      {t.escalated_to_stage_2:,} (Suspect subset)")
    print(f"    - STAGE 2 REDUCTION RATIO:   {t.stage_2_reduction_ratio_pct:.2f}% (Target: > 70%)")

    print("\n[3] FIRST-ALERT SEQUENCE MARKER (Early-Warning Lead Time)")
    print(f"    - First Alert Row ID:        {t.first_alert_row_id}")
    print(f"    - First Alert Sequence:      Turn {t.first_alert_seq_index}")
    print(f"    - EARLY-WARNING LEAD TIME:   {t.early_warning_lead_time_pct:.4f}%")

    print("\n[4] FLAGGED ID SET (Attribution Volume)")
    print(f"    - Total Unique Flagged IDs:  {t.flagged_ids_count:,}")
    print(f"    - Total Tainted Assets Held: {t.tainted_assets_count:,}")

    print("\n[5] RULE ATTRIBUTION BREAKDOWN (Ablation Frequency Table)")
    print(f"    {'Rule Identifier':<38} | {'Trip Count':>10}")
    print("    " + "-" * 52)
    for rule, count in t.rule_attribution_counts.items():
        print(f"    {rule:<38} | {count:>10,}")
    print("=" * 80)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SwarmForensics: Early Checker Telemetry & Ablation Benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/redacted.jsonl",
        help="Path to the uncompressed .jsonl dataset file (default: dataset/redacted.jsonl)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of records to stream. Set to 0 or omit to run on the ENTIRE dataset (default: 0)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["FULL_DPI", "EGRESS_ONLY", "BOTH"],
        default="BOTH",
        help="Evaluation mode: 'FULL_DPI', 'EGRESS_ONLY', or 'BOTH' (default: BOTH)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataset_file = Path(args.dataset)

    # Validate dataset existence
    if not dataset_file.is_file():
        print(f"[!] Error: Dataset not found at: {dataset_file.resolve()}")
        print("    Ensure 'redacted.jsonl' exists in the dataset/ directory or specify --dataset.")
        sys.exit(1)

    # A limit <= 0 means run on the entire dataset
    limit = args.limit if args.limit > 0 else None
    limit_label = f"{limit:,} records" if limit is not None else "ENTIRE DATASET (All rows)"

    print("=" * 80)
    print("STARTING EARLY CHECKER BENCHMARK SUITE")
    print("=" * 80)
    print(f"[*] Target Dataset:  {dataset_file.resolve()}")
    print(f"[*] Ingestion Limit: {limit_label}")
    print(f"[*] Selected Mode:   {args.mode}")

    # Determine which modes to run
    modes_to_run = ["FULL_DPI", "EGRESS_ONLY"] if args.mode == "BOTH" else [args.mode]

    for run_mode in modes_to_run:
        print(f"\n>>> Running Benchmark Mode: [{run_mode}] ...")
        telemetry = run_earlychecker_benchmark(
            dataset_path=dataset_file,
            mode=run_mode,
            limit=limit
        )
        print_telemetry_report(telemetry)


if __name__ == "__main__":
    main()

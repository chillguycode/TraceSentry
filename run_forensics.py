"""
run_forensics.py
Command-Line Interface (CLI) runner for the SwarmForensics pipeline.
Usage:
    python run_forensics.py --limit 5000
    python run_forensics.py --limit 0   # Process all 189,000+ rows
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

from src.parser import SwarmTracesParser
from src.graph_builder import SwarmGraphBuilder
from src.cascade_tracer import isolate_patient_zeros
from src.visualiser import plot_forensic_cluster


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SwarmForensics: Reconstruction and Patient Zero Isolation Engine"
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
        default=5000,
        help="Number of records to stream. Set to 0 or negative to process the entire dataset (default: 5000)"
    )
    parser.add_argument(
        "--min-blast",
        type=int,
        default=1,
        help="Minimum blast radius (lineage + stigmergic consumers) to qualify as Patient Zero (default: 1)"
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="assets/graph.png",
        help="Output filepath for the forensic cluster diagram (default: assets/graph.png)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataset_path = Path(args.dataset)

    # 1. Validate dataset existence
    if not dataset_path.is_file():
        print(f"[!] Error: Dataset file not found at: {dataset_path.resolve()}")
        print("    Please place 'redacted.jsonl' in the dataset/ directory or specify with --dataset.")
        sys.exit(1)

    # A limit <= 0 means 'read the whole file'
    limit = args.limit if args.limit > 0 else None
    limit_str = f"{limit:,} records" if limit is not None else "ENTIRE DATASET (All rows)"

    print("=" * 75)
    print("SWARM FORENSICS: MULTI-AGENT COLLUSION INVESTIGATION")
    print("=" * 75)
    print(f"[*] Target Dataset:    {dataset_path.resolve()}")
    print(f"[*] Processing Limit:  {limit_str}")
    print(f"[*] Min Blast Radius:  {args.min_blast}")

    start_time = time.time()

    # 2. Stream and Parse events line-by-line
    print("\n[*] [Stage 1/3] Streaming and parsing forensic artifacts...")
    stream_generator = SwarmTracesParser.stream_file(dataset_path, limit=limit)

    # 3. Build Tripartite Newman Provenance Graph
    print("[*] [Stage 2/3] Constructing Tripartite Diamond-Newman Graph...")
    builder = SwarmGraphBuilder()
    G = builder.build_graph(stream_generator)

    build_time = time.time() - start_time
    summary = SwarmGraphBuilder.get_graph_summary(G)
    print(f"[+] Graph constructed in {build_time:.2f}s:")
    print(f"    - Total Nodes:     {summary['total_nodes']:,}")
    print(f"    - Total Edges:     {summary['total_edges']:,}")
    print(f"    - Payloads:        {summary['payload_nodes']:,}")
    print(f"    - Infrastructures: {summary['infrastructure_nodes']['total']:,}")
    print(f"    - Capabilities:    {summary['capability_nodes']['total']:,}")

    # 4. Isolate Patient Zeros & Measure Blast Radius
    print("\n[*] [Stage 3/3] Running Forensic Filter (Artifact Novelty + Capability + Blast Radius)...")
    patient_zeros = isolate_patient_zeros(G, min_blast_radius=args.min_blast)

    print(f"[+] Total Suspicious Patient Zeros Identified: {len(patient_zeros)}\n")

    # 5. Display Formatted Forensic Findings Table
    print("=" * 75)
    print("RANKED SWARM ATTRIBUTION REPORT")
    print("=" * 75)

    for rank, pz in enumerate(patient_zeros[:5], start=1):
        print(f"[{rank}] PATIENT ZERO: {pz['patient_zero_candidate']} (Sequence Index: {pz['seq_index']})")
        print(f"    - Classification:       {pz['tier']}")
        print(f"    - Composite Score:      {pz['composite_score']}")
        print(f"    - Total Blast Radius:   {pz['total_blast_radius']} payloads infected")
        print(f"      * Lineage Children:   {pz['lineage_descendants_count']}")
        print(f"      * Stigmergic Readers: {pz['stigmergic_consumers_count']}")

        # Report Capabilities
        if pz["capabilities"]:
            caps_str = ", ".join([f"{c['subtype']} ('{c['raw_match']}')" for c in pz["capabilities"]])
            print(f"    - Exploits Executed:    [{caps_str}]")
        else:
            print(f"    - Exploits Executed:    None (Pure Asset Propagation)")

        # Report Originated Artifacts
        if pz["originated_artifacts"]:
            print(f"    - Originated Loot/Assets ({pz['originated_artifacts_count']}):")
            for art in pz["originated_artifacts"][:3]:
                print(f"      * [{art['subtype']}] {art['value'][:35]} -> Inherited by {art['consumers_count']} payloads")

        print(f"    - Sample Downstream Blast Radius: {pz['sample_descendants']}")
        print("-" * 75)

    # 6. Render and Export Visualization
    if patient_zeros and args.save_plot:
        top_suspect = patient_zeros[0]["patient_zero_candidate"]
        print(f"\n[*] Rendering 2-hop provenance cluster around top suspect: {top_suspect}...")
        plot_forensic_cluster(G, center_node=top_suspect, radius=2, save_path=args.save_plot)

    total_time = time.time() - start_time
    print(f"\n[✔] Forensic pipeline completed successfully in {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()

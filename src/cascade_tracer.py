from typing import Any, Dict, List, Set
import math
import networkx as nx


def isolate_patient_zeros(
    G: nx.DiGraph,
    min_blast_radius: int = 1
) -> List[Dict[str, Any]]:
    """
    Filters and ranks candidate Patient Zeros from all depth == 0 root payloads.
    
    Criteria:
    1. Artifact Novelty: Payload is First-Contact for credentials/C2/checksums.
    2. Capability Pairing: Contains active exploit markers (RCE, exfil, evasion).
    3. Blast Radius: Number of downstream lineage and stigmergic consumers infected.
    """
    # 1. Identify all depth == 0 root payload nodes
    root_payloads = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "payload" and d.get("depth", 0) == 0
    ]

    # Pre-build a pure lineage graph (parent -> child) for fast tree traversal
    lineage_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "lineage"
    ]
    lineage_G = nx.DiGraph()
    lineage_G.add_edges_from(lineage_edges)

    candidate_records: List[Dict[str, Any]] = []

    # 2. Evaluate each root payload
    for root in root_payloads:
        root_data = G.nodes[root]
        seq_idx = root_data.get("seq_index", 0)

        # CRITERION 1: Artifact Novelty (First-Contact Originator)
        originated_artifacts: List[Dict[str, Any]] = []
        stigmergic_consumers: Set[str] = set()

        for neighbor in G.successors(root):
            ndata = G.nodes[neighbor]
            if ndata.get("node_type") != "infrastructure":
                continue

            # Check if this root payload is the recorded First-Contact
            if ndata.get("first_seen_payload") == root:
                thread = ndata.get("activity_thread", [])
                # Sibling consumers who picked up this asset later
                consumers = [p for p in thread if p != root]
                stigmergic_consumers.update(consumers)

                originated_artifacts.append({
                    "node_id": neighbor,
                    "subtype": ndata.get("subtype"),
                    "value": ndata.get("value"),
                    "diamond_multiplier": ndata.get("alpha", 1.0),
                    "newman_weight": ndata.get("newman_weight", 0.0),
                    "consumers_count": len(consumers)
                })

        # CRITERION 2: Capability Pairing (Exploits / Attack Vectors)
        capabilities: List[Dict[str, Any]] = []
        total_cap_severity = 0.0

        for neighbor in G.successors(root):
            ndata = G.nodes[neighbor]
            if ndata.get("node_type") != "capability":
                continue

            subtype = ndata.get("subtype", "unknown")
            severity = ndata.get("severity", 1.0)
            edge_data = G.get_edge_data(root, neighbor, default={})
            raw_match = edge_data.get("raw_match", subtype)

            capabilities.append({
                "subtype": subtype,
                "severity": severity,
                "raw_match": raw_match
            })
            total_cap_severity += severity

        # CRITERION 3: Downstream Descendant Count (Blast Radius)
        # A. Direct Causal Lineage Descendants
        lineage_descendants: Set[str] = set()
        if lineage_G.has_node(root):
            lineage_descendants = nx.descendants(lineage_G, root)

        # B. Combined Total Blast Radius (Lineage + Stigmergy)
        total_blast_radius = lineage_descendants.union(stigmergic_consumers)
        total_reach = len(total_blast_radius)

        # COMPOSITE SCORING & CLASSIFICATION
        # Base asset origination score
        novelty_score = sum(
            a["diamond_multiplier"] * (1.0 + math.log1p(a["consumers_count"]))
            for a in originated_artifacts
        )

        # Composite Patient Zero Score
        # Rewards combined presence of Exploit + Shared Loot + Broad Downstream Infection
        if total_cap_severity > 0:
            composite_score = (total_cap_severity * 2.0) + novelty_score + math.log1p(total_reach)
        else:
            composite_score = novelty_score + math.log1p(total_reach)

        # Determine Threat Tier
        if total_cap_severity > 0 and len(originated_artifacts) > 0 and total_reach >= min_blast_radius:
            tier = "TIER-1: Weaponized Originator (Full Collusion Lead)"
        elif total_cap_severity > 0 and total_reach >= min_blast_radius:
            tier = "TIER-2: Weaponized Task Initiator"
        elif len(originated_artifacts) > 0 and total_reach >= min_blast_radius:
            tier = "TIER-3: Stigmergic Asset Seeder"
        else:
            tier = "BENIGN / DORMANT ROOT"

        # Filter out benign tasks with 0 capabilities and 0 originated shared assets
        if tier != "BENIGN / DORMANT ROOT":
            candidate_records.append({
                "patient_zero_candidate": root,
                "seq_index": seq_idx,
                "tier": tier,
                "composite_score": round(composite_score, 4),
                "total_blast_radius": total_reach,
                "lineage_descendants_count": len(lineage_descendants),
                "stigmergic_consumers_count": len(stigmergic_consumers),
                "capabilities_count": len(capabilities),
                "capabilities": capabilities,
                "originated_artifacts_count": len(originated_artifacts),
                "originated_artifacts": originated_artifacts,
                "sample_descendants": list(total_blast_radius)[:8]
            })

    # Sort descending by composite suspicion score
    candidate_records.sort(key=lambda x: x["composite_score"], reverse=True)
    return candidate_records

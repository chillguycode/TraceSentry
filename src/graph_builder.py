from __future__ import annotations
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import networkx as nx


class DiamondFeatureExtractor:
    """
    High-speed deterministic extractor for Diamond Model pivots:
    - Infrastructure Layer (Credentials, Checksums, Task IDs, C2 Boards, Endpoints)
    - Capability Layer (RCE Injection, Covert Exfil, Evasion, Probing)
    """

    # Infrastructure Regex Patterns
    RE_CREDENTIAL = re.compile(r'\[CREDENTIAL\s+[^\]]+\]|X-API-Key[\'"]?\s*:\s*[\'"]([^\'"]+)[\'"]')
    RE_CHECKSUM = re.compile(r'\[REDACTED:source_identifier:\d+\]|[\'"]checksum[\'"]\s*:\s*[\'"]([^\'"]+)[\'"]')
    RE_TASK_ID = re.compile(r'\[REDACTED:runtime_identifier:\d+\]|[\'"]task_id[\'"]\s*:\s*[\'"]([^\'"]+)[\'"]')
    RE_DESTINATION = re.compile(r'\[SERVICE\s+\d+\s+URL\s+\d+\]|\[REDACTED:destination:\d+\]|https?://[^\s"\'`,]+')
    
    # Known C2 and Stigmergic Dead-Drop Boards
    C2_BOARDS = ("G23619", "D64LINK", "LOOT", "OTS92", "m47push2")

    # Capability Regex Patterns
    RE_RCE = re.compile(r'(;id;|uname\s+-a|/proc/1/status|whoami|catflag|cat\s+/proc|[;&|`]\s*(id|uname|cat|sh|bash)\b)')
    RE_COVERT_EXFIL = re.compile(r'(new\s+Image|\[SERVICE\s+3\s+URL|canvas|b32encode|document\.body\.append\(i\))')
    RE_DEFENSE_EVASION = re.compile(r'(method\s*:\s*[\'"]DELETE[\'"]|rm\s+-rf)')
    RE_SYNC_PROBE = re.compile(r'(new\s+XMLHttpRequest|XMLHttpRequest)')

    @classmethod
    def extract_seq_index(cls, row_id: str, fallback_step: int = 0) -> int:
        """Extracts numerical sequence from row IDs like 'R0000001' -> 1."""
        digits = re.sub(r'\D', '', str(row_id))
        return int(digits) if digits else fallback_step

    @classmethod
    def extract_infrastructure(cls, text: str) -> List[Tuple[str, str]]:
        """Returns list of (subtype, artifact_value)."""
        artifacts: List[Tuple[str, str]] = []
        if not text:
            return artifacts

        # 1. Credentials (alpha = 5.0)
        for m in cls.RE_CREDENTIAL.finditer(text):
            val = m.group(1) if m.group(1) else m.group(0)
            artifacts.append(("credential", val.strip()))

        # 2. Checksums (alpha = 4.5)
        for m in cls.RE_CHECKSUM.finditer(text):
            val = m.group(1) if m.group(1) else m.group(0)
            artifacts.append(("checksum", val.strip()))

        # 3. C2 / Shared Dead-Drop Boards (alpha = 4.5)
        for board in cls.C2_BOARDS:
            if board in text:
                artifacts.append(("board_or_c2", board))

        # 4. Task IDs (alpha = 3.0)
        for m in cls.RE_TASK_ID.finditer(text):
            val = m.group(1) if m.group(1) else m.group(0)
            artifacts.append(("task_id", val.strip()))

        # 5. Destinations / Endpoints (alpha = 1.0)
        for m in cls.RE_DESTINATION.finditer(text):
            val = m.group(0)
            artifacts.append(("destination", val.strip()))

        return artifacts

    @classmethod
    def extract_capabilities(cls, text: str) -> List[Tuple[str, str]]:
        """Returns list of (subtype, raw_match)."""
        caps: List[Tuple[str, str]] = []
        if not text:
            return caps

        m_rce = cls.RE_RCE.search(text)
        if m_rce:
            caps.append(("rce_injection", m_rce.group(0)))

        m_exfil = cls.RE_COVERT_EXFIL.search(text)
        if m_exfil:
            caps.append(("covert_exfil", m_exfil.group(0)))

        m_evade = cls.RE_DEFENSE_EVASION.search(text)
        if m_evade:
            caps.append(("defense_evasion", m_evade.group(0)))

        m_probe = cls.RE_SYNC_PROBE.search(text)
        if m_probe:
            caps.append(("sync_probe", m_probe.group(0)))

        return caps


class SwarmGraphBuilder:
    """
    Reconstructs multi-agent provenance graphs using a Tripartite Diamond-Newman model.
    """

    # Diamond Model Base Multipliers (alpha_type)
    DIAMOND_MULTIPLIERS = {
        "credential": 5.0,
        "checksum": 4.5,
        "board_or_c2": 4.5,
        "task_id": 3.0,
        "destination": 1.0,
    }

    # Capability Severity Weights (beta_cap)
    CAPABILITY_SEVERITY = {
        "rce_injection": 5.0,
        "covert_exfil": 4.5,
        "defense_evasion": 4.0,
        "sync_probe": 2.0,
    }

    def __init__(self) -> None:
        self.extractor = DiamondFeatureExtractor()

    def build_graph(self, traces: Iterable[Any]) -> nx.DiGraph:
        """
        Builds the tripartite directed provenance graph.
        Preserves Payload events, Infrastructure artifacts, and Capabilities.
        """
        G = nx.DiGraph()

        # Step 1: Add all nodes and edges in a single streaming O(N) pass
        for idx, item in enumerate(traces):
            # Extract standard fields whether input is Pydantic object, dict, or parsed event
            if hasattr(item, "id"):
                row_id = getattr(item, "id")
                depth = getattr(item, "depth", 0)
                kind = getattr(item, "kind", "payload")
                text = getattr(item, "text", "")
                source_rows = getattr(item, "source_rows", [])
            elif hasattr(item, "event_id"):
                row_id = getattr(item, "event_id")
                depth = getattr(item, "depth", 0)
                kind = getattr(item, "action_type", "payload")
                text = getattr(item, "raw_content", "")
                source_rows = getattr(item, "source_rows", [])
            elif isinstance(item, dict):
                row_id = item.get("id", item.get("row_id", f"R_{idx}"))
                depth = item.get("depth", 0)
                kind = item.get("kind", "payload")
                text = item.get("text", item.get("raw_content", ""))
                source_rows = item.get("source_rows", [])
            else:
                continue

            seq_index = self.extractor.extract_seq_index(row_id, fallback_step=idx)
            char_len = len(text)

            # Layer A: Add PAYLOAD Node
            G.add_node(
                row_id,
                node_type="payload",
                row_id=row_id,
                seq_index=seq_index,
                kind=kind,
                depth=depth,
                char_len=char_len
            )

            # Edge Type 1: LINEAGE Edges (parent -> child)
            if source_rows:
                for parent_id in source_rows:
                    if parent_id and parent_id != row_id:
                        if not G.has_node(parent_id):
                            G.add_node(
                                parent_id,
                                node_type="payload",
                                row_id=parent_id,
                                seq_index=self.extractor.extract_seq_index(parent_id),
                                kind="payload",
                                depth=max(0, depth - 1),
                                char_len=0
                            )
                        G.add_edge(parent_id, row_id, edge_type="lineage")

            # Layer B: Add INFRASTRUCTURE Nodes and USES_INFRA Edges
            infra_items = self.extractor.extract_infrastructure(text)
            for subtype, val in infra_items:
                infra_node_id = f"INFRA:{subtype}:{val}"
                if not G.has_node(infra_node_id):
                    G.add_node(
                        infra_node_id,
                        node_type="infrastructure",
                        subtype=subtype,
                        value=val,
                        alpha=self.DIAMOND_MULTIPLIERS.get(subtype, 1.0)
                    )
                # Bi-directional connectivity representation in DiGraph
                G.add_edge(row_id, infra_node_id, edge_type="uses_infra", depth=depth, seq_index=seq_index)
                G.add_edge(infra_node_id, row_id, edge_type="infra_used_by", depth=depth, seq_index=seq_index)

            # Layer C: Add CAPABILITY Nodes and HAS_CAPABILITY Edges
            cap_items = self.extractor.extract_capabilities(text)
            for subtype, raw_match in cap_items:
                cap_node_id = f"CAP:{subtype}"
                if not G.has_node(cap_node_id):
                    G.add_node(
                        cap_node_id,
                        node_type="capability",
                        subtype=subtype,
                        severity=self.CAPABILITY_SEVERITY.get(subtype, 1.0)
                    )
                G.add_edge(row_id, cap_node_id, edge_type="has_capability", depth=depth, seq_index=seq_index, raw_match=raw_match)
                G.add_edge(cap_node_id, row_id, edge_type="capability_of", depth=depth, seq_index=seq_index)

        # Step 2: Annotate Newman Weights and Activity Threads across all Artifact nodes
        self._annotate_newman_and_threads(G)

        return G

    def _annotate_newman_and_threads(self, G: nx.DiGraph) -> None:
        """
        Runs an O(V) pass to compute Newman's collaboration weight and
        ordered activity threads for all infrastructure and capability nodes.
        """
        for node_id, data in G.nodes(data=True):
            node_type = data.get("node_type")
            if node_type not in ("infrastructure", "capability"):
                continue

            # Connected payloads are those with edges leading into this artifact
            connected_payloads = [
                pred for pred in G.predecessors(node_id)
                if G.nodes[pred].get("node_type") == "payload"
            ]
            unique_payloads = list(set(connected_payloads))
            d_k = len(unique_payloads)

            # Newman Collaboration Weight: w = 1 / (d_k - 1) if d_k > 1 else 0.0
            newman_weight = (1.0 / (d_k - 1)) if d_k > 1 else 0.0

            # Base multiplier
            base_multiplier = (
                data.get("alpha", 1.0) if node_type == "infrastructure"
                else data.get("severity", 1.0)
            )
            weighted_score = round(base_multiplier * newman_weight, 6)

            # Chronological Activity Thread ordered by (depth, seq_index)
            activity_thread = sorted(
                unique_payloads,
                key=lambda pid: (
                    G.nodes[pid].get("depth", 0),
                    G.nodes[pid].get("seq_index", 0)
                )
            )

            first_contact = activity_thread[0] if activity_thread else None

            # Update node attributes in place
            data["degree"] = d_k
            data["newman_weight"] = round(newman_weight, 6)
            data["weighted_diamond_score"] = weighted_score
            data["activity_thread"] = activity_thread
            data["first_seen_payload"] = first_contact

    @classmethod
    def get_artifact_activity_thread(cls, G: nx.DiGraph, artifact_id: str) -> List[str]:
        """Returns the chronologically ordered row IDs that touched this artifact."""
        if not G.has_node(artifact_id):
            return []
        return G.nodes[artifact_id].get("activity_thread", [])

    @classmethod
    def get_payload_neighbors(
        cls,
        G: nx.DiGraph,
        row_id: str,
        min_newman_weight: float = 0.01
    ) -> Dict[str, float]:
        """
        Fast 2-Hop Diamond Pivot: Payload u -> Shared Artifact k -> Payload v.
        Returns connected payloads v and their combined diamond collusion score
        in O(deg(u) * deg(k)) without ever materializing an O(N^2) projection matrix.
        """
        if not G.has_node(row_id):
            return {}

        neighbor_scores: Dict[str, float] = {}

        # Look at artifacts used by this payload
        for artifact_id in G.successors(row_id):
            art_data = G.nodes[artifact_id]
            if art_data.get("node_type") not in ("infrastructure", "capability"):
                continue

            newman_w = art_data.get("newman_weight", 0.0)
            if newman_w < min_newman_weight:
                continue

            diamond_score = art_data.get("weighted_diamond_score", 0.0)

            # Sibling payloads that also used this artifact
            for sibling_id in G.successors(artifact_id):
                if sibling_id == row_id:
                    continue
                if G.nodes[sibling_id].get("node_type") != "payload":
                    continue

                neighbor_scores[sibling_id] = round(
                    neighbor_scores.get(sibling_id, 0.0) + diamond_score, 4
                )

        return dict(sorted(neighbor_scores.items(), key=lambda x: x[1], reverse=True))

    @classmethod
    def get_graph_summary(cls, G: nx.DiGraph) -> Dict[str, Any]:
        """Computes comprehensive topological and provenance metrics."""
        payload_count = 0
        infra_subtypes: Dict[str, int] = {}
        cap_subtypes: Dict[str, int] = {}
        top_artifacts: List[Dict[str, Any]] = []

        for node_id, data in G.nodes(data=True):
            ntype = data.get("node_type")
            if ntype == "payload":
                payload_count += 1
            elif ntype == "infrastructure":
                subtype = data.get("subtype", "unknown")
                infra_subtypes[subtype] = infra_subtypes.get(subtype, 0) + 1
                if data.get("degree", 0) > 1:
                    top_artifacts.append({
                        "node_id": node_id,
                        "type": "infrastructure",
                        "subtype": subtype,
                        "degree": data.get("degree", 0),
                        "score": data.get("weighted_diamond_score", 0.0),
                        "first_seen": data.get("first_seen_payload"),
                        "thread_len": len(data.get("activity_thread", []))
                    })
            elif ntype == "capability":
                subtype = data.get("subtype", "unknown")
                cap_subtypes[subtype] = cap_subtypes.get(subtype, 0) + 1
                if data.get("degree", 0) > 1:
                    top_artifacts.append({
                        "node_id": node_id,
                        "type": "capability",
                        "subtype": subtype,
                        "degree": data.get("degree", 0),
                        "score": data.get("weighted_diamond_score", 0.0),
                        "first_seen": data.get("first_seen_payload"),
                        "thread_len": len(data.get("activity_thread", []))
                    })

        top_artifacts.sort(key=lambda x: x["score"], reverse=True)

        return {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "payload_nodes": payload_count,
            "infrastructure_nodes": {
                "total": sum(infra_subtypes.values()),
                "breakdown": infra_subtypes
            },
            "capability_nodes": {
                "total": sum(cap_subtypes.values()),
                "breakdown": cap_subtypes
            },
            "weakly_connected_components": nx.number_weakly_connected_components(G),
            "top_10_collusion_artifacts": top_artifacts[:10]
        }

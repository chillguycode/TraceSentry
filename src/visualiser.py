from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import math
import hashlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

FONT_FAMILY = "STIX Two Text"
plt.rcParams["font.family"] = FONT_FAMILY

# Publication-oriented visual settings
FIGURE_SIZE = (14.5, 10.0)
FIGURE_DPI = 180
SAVE_DPI = 300

ROOT_COLOR = "#D64545"
PAYLOAD_COLOR = "#4A9EDB"
INFRA_COLOR = "#E6A23C"
CAPABILITY_COLOR = "#8E63B5"
LINEAGE_EDGE_COLOR = "#9FA8B2"
ARTIFACT_EDGE_COLOR = "#D98D13"
CAPABILITY_EDGE_COLOR = "#8E63B5"
TEXT_COLOR = "#20252B"
WHITE = "#FFFFFF"
GRID_COLOR = "#E6EAF0"


# Graph helpers
def _node_type(G: nx.Graph, node) -> str:
    value = str(G.nodes[node].get("node_type", "payload")).strip().lower()
    if value in {"infra", "infrastructure", "artifact", "resource", "loot"}:
        return "infrastructure"
    if value in {"cap", "capability", "attack_capability"}:
        return "capability"
    return "payload"


def _short_value(value, limit: int = 28) -> str:
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _edge_relation(G: nx.Graph, source, target, data: dict) -> str:
    raw = ""
    for key in ("edge_type", "relation", "relationship", "kind", "type"):
        if data.get(key) is not None:
            raw = str(data[key]).lower()
            break

    if any(t in raw for t in ("lineage", "parent", "child")):
        return "lineage"
    if any(t in raw for t in ("capability", "attack_cap")):
        return "capability"
    if any(t in raw for t in ("artifact", "infrastructure", "resource", "uses")):
        return "artifact"

    st, tt = _node_type(G, source), _node_type(G, target)
    if st == "capability" or tt == "capability":
        return "capability"
    if st == "infrastructure" or tt == "infrastructure":
        return "artifact"
    if st == "payload" and tt == "payload":
        return "lineage"
    return "other"


def _iter_edges(G: nx.Graph) -> Iterable[Tuple[object, object, dict]]:
    if G.is_multigraph():
        for u, v, _, data in G.edges(keys=True, data=True):
            yield u, v, data
    else:
        for u, v, data in G.edges(data=True):
            yield u, v, data


def _stable_order(nodes: Iterable, G: nx.Graph) -> list:
    def key(node):
        data = G.nodes[node]
        seq = data.get("sequence", data.get("seq", data.get("turn", float("inf"))))
        try:
            seq = float(seq)
        except (TypeError, ValueError):
            seq = float("inf")
        return seq, str(node)

    return sorted(nodes, key=key)


def _stable_float(node, salt: str = "0") -> float:
    raw = f"{salt}|{node}".encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    integer = int.from_bytes(digest, "big")
    return integer / 2**64


def _artifact_neighbors(G: nx.Graph, subgraph: nx.Graph, node) -> List:
    return [
        n
        for n in subgraph.neighbors(node)
        if _node_type(G, n) == "infrastructure"
        and _edge_relation(G, node, n, subgraph.get_edge_data(node, n) or {}) == "artifact"
    ]


def _shared_state_consumers(G: nx.Graph, subgraph: nx.Graph, root) -> List:
    root_infra = {
        n for n in subgraph.neighbors(root) if _node_type(G, n) == "infrastructure"
    }
    if not root_infra:
        return []

    consumers = []
    for node in subgraph:
        if node == root or _node_type(G, node) != "payload":
            continue
        if any(artifact in subgraph.neighbors(node) for artifact in root_infra):
            consumers.append(node)
    return consumers


# Layout
def _core_positions(infra: List, capabilities: List, root) -> Dict:
    """Place the root and infrastructure in a clean forensic core."""
    pos = {root: (0.0, 3.95)}

    n = len(infra)
    if n:
        # Slightly curved horizontal band. This leaves a clean receiving area
        # underneath each infrastructure node for its connected payloads.
        x_span = min(5.2, max(2.6, 0.82 * (n - 1)))
        if n == 1:
            x_values = [0.0]
        else:
            x_values = [(-x_span / 2.0) + x_span * i / (n - 1) for i in range(n)]

        for i, node in enumerate(infra):
            x = x_values[i]
            y = 1.75 - 0.12 * (abs(x) / max(1.0, x_span / 2.0))
            pos[node] = (x, y)

    if capabilities:
        total = len(capabilities)
        cap_span = min(4.6, max(2.0, 0.85 * (total - 1)))
        for i, node in enumerate(capabilities):
            if total == 1:
                x = 0.0
            else:
                x = -cap_span / 2.0 + cap_span * i / (total - 1)
            pos[node] = (x, 0.55)

    return pos


def _sector_angles(infra: List) -> Dict:
    """Give each infrastructure node an angular sector around the payload field."""
    if not infra:
        return {}

    # The fan occupies a broad lower hemisphere. More central artifacts get a
    # slightly wider visual neighborhood because they tend to have more overlap.
    start = math.radians(165)
    end = math.radians(15)
    n = len(infra)

    if n == 1:
        return {infra[0]: math.radians(90)}

    return {
        node: start + (end - start) * i / (n - 1)
        for i, node in enumerate(infra)
    }


def _payload_positions(
    G: nx.Graph,
    subgraph: nx.Graph,
    payloads: List,
    infra: List,
    center: Tuple[float, float] = (0.0, -0.8),
    inner_radius: float = 1.75,
    outer_radius: float = 5.45,
) -> Dict:
    """Place payloads in artifact-aware sectors while keeping them deterministic.

    The critical idea is that payloads touching the same infrastructure are
    spatially adjacent. Their exact payload↔artifact edges remain visible, so
    the layout itself helps the reader follow provenance without sampling edges.
    """
    if not payloads:
        return {}

    pos: Dict = {}
    sector_angles = _sector_angles(infra)

    grouped: Dict[object, List] = {artifact: [] for artifact in infra}
    unassigned: List = []

    for node in payloads:
        touched = _artifact_neighbors(G, subgraph, node)
        if not touched:
            unassigned.append(node)
            continue

        # Primary sector: artifact with the strongest local importance.
        primary = max(touched, key=lambda a: (subgraph.degree(a), str(a)))
        grouped.setdefault(primary, []).append(node)

    # If a payload touches multiple artifacts, position it near the circular
    # mean of those artifacts rather than forcing it into only one sector.
    for artifact, nodes in grouped.items():
        if not nodes:
            continue

        angle = sector_angles.get(artifact, math.radians(90))
        # Sort by stable sequence so the outer field looks coherent between runs.
        nodes = _stable_order(nodes, G)
        m = len(nodes)

        # A golden-angle fill within the sector gives readable spacing even for
        # several thousand consumers.
        local_spread = math.radians(max(7.0, min(26.0, 230.0 / max(1, len(infra)))))
        golden = math.pi * (3.0 - math.sqrt(5.0))

        for i, node in enumerate(nodes):
            frac = (i + 0.5) / m
            radius = inner_radius + (outer_radius - inner_radius) * math.sqrt(frac)

            jitter = (_stable_float(node, "angle") - 0.5) * local_spread
            theta = angle + jitter + 0.12 * math.sin(i * golden)

            # Multi-artifact payloads get a small deterministic inward bias,
            # visually tying them to the shared central core.
            touched = _artifact_neighbors(G, subgraph, node)
            inward = min(0.34, 0.08 * max(0, len(touched) - 1))
            radius = max(inner_radius, radius - inward)

            pos[node] = (
                center[0] + radius * math.cos(theta),
                center[1] + 0.83 * radius * math.sin(theta),
            )

    # Payloads that do not touch infrastructure are placed in a quiet bottom
    # arc. They are still fully represented but don't compete with the main
    # stigmergic structure.
    if unassigned:
        unassigned = _stable_order(unassigned, G)
        m = len(unassigned)
        for i, node in enumerate(unassigned):
            frac = (i + 0.5) / m
            theta = math.radians(205 + 130 * frac)
            radius = outer_radius * (0.88 + 0.10 * frac)
            pos[node] = (
                center[0] + radius * math.cos(theta),
                center[1] + 0.83 * radius * math.sin(theta),
            )

    return pos


# Labels / edge extraction
def _select_labels(
    G: nx.Graph,
    subgraph: nx.Graph,
    center_node,
    max_infrastructure_labels: int = 10,
    max_capability_labels: int = 4,
    max_payload_labels: int = 6,
):
    labels = {center_node: str(center_node)}

    infra = [n for n in subgraph if _node_type(G, n) == "infrastructure"]
    infra = sorted(infra, key=lambda n: (-subgraph.degree(n), str(n)))
    infra_labeled = infra[:max_infrastructure_labels]
    for node in infra_labeled:
        d = G.nodes[node]
        subtype = _short_value(d.get("subtype", "artifact"), 20)
        value = _short_value(str(d.get("value", node)).split("/")[-1], 24)
        labels[node] = f"{subtype}\n{value}"

    caps = [n for n in subgraph if _node_type(G, n) == "capability"]
    caps = sorted(caps, key=lambda n: (-subgraph.degree(n), str(n)))
    caps_labeled = caps[:max_capability_labels]
    for node in caps_labeled:
        labels[node] = f"CAP: {_short_value(G.nodes[node].get('subtype', node), 24)}"

    payloads = [n for n in subgraph if _node_type(G, n) == "payload" and n != center_node]
    payloads = _stable_order(payloads, G)

    # Prefer payloads that touch multiple artifacts: those are especially
    # informative because they visibly bridge infrastructure sectors.
    multi = [n for n in payloads if len(_artifact_neighbors(G, subgraph, n)) >= 2]
    selected: List = []
    selected.extend(multi[: max(0, max_payload_labels // 2)])

    for node in payloads:
        if len(selected) >= max_payload_labels:
            break
        if node not in selected:
            selected.append(node)

    selected = selected[:max_payload_labels]
    for node in selected:
        labels[node] = str(node)

    return labels, infra_labeled, caps_labeled, selected


def _edge_buckets(G: nx.Graph, subgraph: nx.Graph):
    artifact_edges = []
    lineage_edges = []
    capability_edges = []

    for u, v, data in _iter_edges(subgraph):
        rel = _edge_relation(G, u, v, data)
        if rel == "artifact":
            artifact_edges.append((u, v))
        elif rel == "lineage":
            lineage_edges.append((u, v))
        elif rel == "capability":
            capability_edges.append((u, v))

    return artifact_edges, lineage_edges, capability_edges


def _draw_edges(
    G: nx.Graph,
    subgraph: nx.Graph,
    pos: Dict,
    root,
    ax,
    artifact_edge_alpha: float = 0.13,
    artifact_edge_width: float = 0.45,
    lineage_edge_alpha: float = 0.26,
    lineage_edge_width: float = 0.62,
    capability_edge_alpha: float = 0.46,
    show_lineage: bool = True,
):
    artifact_edges, lineage_edges, capability_edges = _edge_buckets(G, subgraph)

    # FIRST: all payload↔artifact relationships are drawn. No fan-out sampling.
    # Root→artifact edges are then redrawn more strongly on top.
    if artifact_edges:
        nx.draw_networkx_edges(
            subgraph,
            pos,
            edgelist=artifact_edges,
            edge_color=ARTIFACT_EDGE_COLOR,
            alpha=artifact_edge_alpha,
            width=artifact_edge_width,
            arrows=False,
            ax=ax,
        )

    root_artifact = [e for e in artifact_edges if root in e]
    if root_artifact:
        nx.draw_networkx_edges(
            subgraph,
            pos,
            edgelist=root_artifact,
            edge_color=ARTIFACT_EDGE_COLOR,
            alpha=0.82,
            width=1.65,
            arrows=False,
            ax=ax,
        )

    if show_lineage and lineage_edges:
        nx.draw_networkx_edges(
            subgraph,
            pos,
            edgelist=lineage_edges,
            edge_color=LINEAGE_EDGE_COLOR,
            alpha=lineage_edge_alpha,
            width=lineage_edge_width,
            arrows=False,
            ax=ax,
        )

    if capability_edges:
        nx.draw_networkx_edges(
            subgraph,
            pos,
            edgelist=capability_edges,
            edge_color=CAPABILITY_EDGE_COLOR,
            alpha=capability_edge_alpha,
            width=0.75,
            style=(0, (3, 3)),
            arrows=False,
            ax=ax,
        )

    return artifact_edges, lineage_edges, capability_edges


# Main public renderer
def plot_forensic_cluster(
    G: nx.DiGraph,
    center_node: str = "R0000001",
    radius: int = 2,
    save_path: Optional[str] = None,
    show_title: bool = True,
    max_infrastructure_labels: int = 10,
    max_capability_labels: int = 4,
    max_payload_labels: int = 6,
    artifact_edge_alpha: float = 0.13,
    artifact_edge_width: float = 0.45,
    show_lineage: bool = True,
) -> None:
    """Render a publication-oriented 2-hop provenance neighborhood.

    Design goals
    ------------
    1. Keep the complete 2-hop graph extraction unchanged.
    2. Preserve *all* payload↔artifact relationships in the rendered figure.
    3. Arrange payloads by the infrastructure they touch so provenance edges
       remain readable even in 5,000+ node fan-out clusters.
    4. Keep labels selective: the graph is complete, the text is not.

    The figure therefore communicates both scale and the forensic question
    "which events touched which shared artifact?" without requiring thousands
    of event-ID labels.
    """
    if not G.has_node(center_node):
        print(f"[-] Node {center_node} not found in graph.")
        return
    if radius < 0:
        raise ValueError("radius must be >= 0")
    if artifact_edge_alpha < 0 or artifact_edge_alpha > 1:
        raise ValueError("artifact_edge_alpha must be between 0 and 1")
    if artifact_edge_width <= 0:
        raise ValueError("artifact_edge_width must be > 0")

    undirected = G.to_undirected(as_view=True) if G.is_directed() else G
    subgraph = nx.ego_graph(undirected, center_node, radius=radius)

    payload_nodes = [n for n in subgraph if _node_type(G, n) == "payload"]
    infra_nodes = sorted(
        [n for n in subgraph if _node_type(G, n) == "infrastructure"],
        key=lambda n: (-subgraph.degree(n), str(n)),
    )
    capability_nodes = sorted(
        [n for n in subgraph if _node_type(G, n) == "capability"],
        key=lambda n: (-subgraph.degree(n), str(n)),
    )
    payload_without_root = [n for n in payload_nodes if n != center_node]

    pos = {}
    pos.update(_core_positions(infra_nodes, capability_nodes, center_node))
    pos.update(
        _payload_positions(
            G,
            subgraph,
            _stable_order(payload_without_root, G),
            infra_nodes,
        )
    )

    labels, infra_labeled, caps_labeled, payload_labeled = _select_labels(
        G,
        subgraph,
        center_node,
        max_infrastructure_labels=max_infrastructure_labels,
        max_capability_labels=max_capability_labels,
        max_payload_labels=max_payload_labels,
    )

    shared_consumers = _shared_state_consumers(G, subgraph, center_node)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    artifact_edges, lineage_edges, capability_edges = _draw_edges(
        G,
        subgraph,
        pos,
        center_node,
        ax,
        artifact_edge_alpha=artifact_edge_alpha,
        artifact_edge_width=artifact_edge_width,
        show_lineage=show_lineage,
    )

    # Payload population: visible enough to preserve scale, quiet enough to let
    # the orange provenance edges remain readable.
    if payload_without_root:
        # Tiny nodes work well at 6k+ scale; enlarge modestly for small graphs.
        payload_size = 16 if len(payload_without_root) > 2500 else 26
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            nodelist=payload_without_root,
            node_color=PAYLOAD_COLOR,
            node_size=payload_size,
            alpha=0.77,
            edgecolors=WHITE,
            linewidths=0.42,
            ax=ax,
        )

    if infra_nodes:
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            nodelist=infra_nodes,
            node_color=INFRA_COLOR,
            node_size=430 if len(infra_nodes) <= 8 else 350,
            alpha=0.99,
            edgecolors=WHITE,
            linewidths=1.7,
            ax=ax,
        )

    if capability_nodes:
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            nodelist=capability_nodes,
            node_color=CAPABILITY_COLOR,
            node_size=290,
            alpha=0.96,
            edgecolors=WHITE,
            linewidths=1.4,
            ax=ax,
        )

    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=[center_node],
        node_color=ROOT_COLOR,
        node_size=920,
        alpha=1.0,
        edgecolors=WHITE,
        linewidths=2.2,
        ax=ax,
    )

    # Labels
    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels={center_node: labels[center_node]},
        font_size=10,
        font_family= FONT_FAMILY,
        font_weight="bold",
        font_color=TEXT_COLOR,
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor=WHITE,
            edgecolor="#D8DDE3",
            linewidth=0.7,
            alpha=0.95,
        ),
        ax=ax,
    )

    # Infrastructure labels use leader lines so adjacent artifact nodes never
    # collide. Offset direction is deterministic and follows the artifact row.
    for i, node in enumerate(infra_labeled):
        x, y = pos[node]
        upward = i % 2 == 0
        dx = (-18 if i % 2 == 0 else 18)
        dy = (28 if upward else -31)
        ha = "right" if dx < 0 else "left"
        ax.annotate(
            labels[node],
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="bottom" if upward else "top",
            fontsize=7.2,
            fontfamily= FONT_FAMILY,
            fontweight="bold",
            color=TEXT_COLOR,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=WHITE,
                edgecolor="#D8DDE3",
                linewidth=0.6,
                alpha=0.94,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#B4BBC4",
                linewidth=0.6,
                shrinkA=4,
                shrinkB=4,
            ),
            zorder=8,
        )

    for i, node in enumerate(caps_labeled):
        x, y = pos[node]
        ax.annotate(
            labels[node],
            xy=(x, y),
            xytext=(0, -27),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.1,
            fontfamily= FONT_FAMILY,
            fontweight="bold",
            color=TEXT_COLOR,
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor=WHITE,
                edgecolor="#D8DDE3",
                linewidth=0.6,
                alpha=0.94,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#B4BBC4",
                linewidth=0.55,
                shrinkA=3,
                shrinkB=3,
            ),
            zorder=8,
        )

    # Label a few representative payloads with leader lines. These are chosen
    # partly from multi-artifact events because they are especially informative.
    for i, node in enumerate(payload_labeled):
        x, y = pos[node]
        angle = math.atan2(y + 0.8, x)
        dx = 18 if math.cos(angle) >= 0 else -18
        dy = 16 if math.sin(angle) >= 0 else -16
        ax.annotate(
            labels[node],
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            fontsize=6.0,
            fontfamily= FONT_FAMILY,
            color="#3B4652",
            bbox=dict(
                boxstyle="round,pad=0.10",
                facecolor=WHITE,
                edgecolor="none",
                alpha=0.68,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#9AA4AF",
                linewidth=0.45,
                shrinkA=2,
                shrinkB=2,
            ),
            zorder=7,
        )

    # Compact annotation and statistics
    ax.text(
        0.5,
        0.085,
        "All payload ↔ shared-artifact relationships shown; labels are selective",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        color="#5A6570",
    )

    stats = [
        f"Payload events: {len(payload_nodes):,}",
        f"Shared-state consumers: {len(shared_consumers):,}",
        f"Infrastructure: {len(infra_nodes):,}",
        f"Capabilities: {len(capability_nodes):,}",
        f"Payload ↔ artifact edges: {len(artifact_edges):,}",
    ]
    ax.text(
        0.012,
        0.018,
        "\n".join(stats),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=TEXT_COLOR,
        linespacing=1.4,
        bbox=dict(
            boxstyle="round,pad=0.42",
            facecolor=WHITE,
            edgecolor="#D8DDE3",
            linewidth=0.8,
            alpha=0.94,
        ),
    )

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=ROOT_COLOR,
            markeredgecolor=WHITE, markeredgewidth=1.2, markersize=9,
            label="Root payload",
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=PAYLOAD_COLOR,
            markeredgecolor=WHITE, markeredgewidth=0.7, markersize=6,
            label="Payload event",
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=INFRA_COLOR,
            markeredgecolor=WHITE, markeredgewidth=1.0, markersize=8,
            label="Shared infrastructure",
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=CAPABILITY_COLOR,
            markeredgecolor=WHITE, markeredgewidth=1.0, markersize=8,
            label="Attack capability",
        ),
        Line2D(
            [0], [0], color=ARTIFACT_EDGE_COLOR, linewidth=1.55,
            label="Root → shared artifact",
        ),
        Line2D(
            [0], [0], color=ARTIFACT_EDGE_COLOR, linewidth=0.7, alpha=0.55,
            label="Payload → shared artifact (all shown)",
        ),
        Line2D(
            [0], [0], color=LINEAGE_EDGE_COLOR, linewidth=0.7,
            label="Lineage / provenance",
        ),
        Line2D(
            [0], [0], color=CAPABILITY_EDGE_COLOR, linewidth=1.0,
            linestyle=(0, (3, 3)), label="Capability association",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor=WHITE,
        edgecolor="#D8DDE3",
        fontsize=8.2,
        borderpad=0.8,
        labelspacing=0.55,
    )

    if show_title:
        ax.set_title(
            f"Two-Hop Provenance Subgraph around {center_node} (Radius = {radius})",
            fontsize=15,
            fontweight="bold",
            color=TEXT_COLOR,
            pad=14,
        )

    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.10)
    fig.tight_layout(pad=1.0)

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_file,
            dpi=SAVE_DPI,
            bbox_inches="tight",
            facecolor=WHITE,
        )
        print(f"[+] Forensic graph successfully saved to: {out_file.resolve()}")
    else:
        plt.show()

    plt.close(fig)


# Optional demo / smoke test
def build_demo_graph(
    payload_count: int = 6576,
    infrastructure_count: int = 10,
    seed: int = 42,
) -> nx.DiGraph:
    """Build a deterministic high-degree demo resembling the user's cluster.

    This is only for testing the renderer. It is not part of the TraceSentry
    forensic data model and should not be used as research data.
    """
    G = nx.DiGraph()
    root = "R005280"

    G.add_node(root, node_type="payload", sequence=1)

    infra_nodes = []
    for i in range(infrastructure_count):
        node = f"I{i+1:03d}"
        infra_nodes.append(node)
        G.add_node(
            node,
            node_type="infrastructure",
            subtype="credential",
            value=f"[CREDENTIAL {169 + i}]",
        )
        G.add_edge(root, node, edge_type="artifact")

    # Nearly every payload is a shared-state consumer. A small fraction touches
    # two artifacts to create visible cross-sector provenance bridges.
    next_id = 1
    for i in range(payload_count - 1):
        while f"R{next_id:06d}" == root:
            next_id += 1
        node = f"R{next_id:06d}"
        G.add_node(node, node_type="payload", sequence=next_id)

        primary = infra_nodes[(i * 7 + seed) % len(infra_nodes)]
        G.add_edge(node, primary, edge_type="artifact")

        if i % 19 == 0:
            secondary = infra_nodes[(i * 11 + seed + 3) % len(infra_nodes)]
            if secondary != primary:
                G.add_edge(node, secondary, edge_type="artifact")

        # A sparse ordinary lineage backbone, so the renderer also demonstrates
        # how ordinary provenance differs from shared-state propagation.
        if i >= 2 and i <= 45:
            parent_id = next_id - 1
            while f"R{parent_id:06d}" == root:
                parent_id -= 1
            parent = f"R{parent_id:06d}"
            if G.has_node(parent):
                G.add_edge(parent, node, edge_type="lineage")

        next_id += 1

    return G

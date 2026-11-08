from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx


def plot_forensic_cluster(
    G: nx.DiGraph,
    center_node: str = "R0000001",
    radius: int = 2,
    save_path: Optional[str] = None
) -> None:
    """
    Renders a fast, color-coded 2-hop neighborhood around a center node.
    If save_path is provided, exports to disk; otherwise calls plt.show().
    """
    if not G.has_node(center_node):
        print(f"[-] Node {center_node} not found in graph.")
        return

    # 1. Extract the local 2-hop neighborhood (to_undirected allows bi-directional traversal)
    subgraph = nx.ego_graph(G.to_undirected(), center_node, radius=radius)
    
    # 2. Assign colors and sizes based on the Diamond node type
    node_colors = []
    node_sizes = []
    labels = {}

    for node in subgraph.nodes():
        node_data = G.nodes[node]
        ntype = node_data.get("node_type", "payload")

        if node == center_node:
            node_colors.append("#e74c3c")  # Crimson Red for Target/Patient Zero
            node_sizes.append(900)
            labels[node] = f"★ {node}"
        elif ntype == "payload":
            node_colors.append("#3498db")  # Sky Blue for other Payloads
            node_sizes.append(450)
            labels[node] = node
        elif ntype == "infrastructure":
            node_colors.append("#f39c12")  # Amber/Orange for Infrastructure/Loot
            node_sizes.append(600)
            val = str(node_data.get("value", node)).split("/")[-1]
            labels[node] = f"[{node_data.get('subtype', 'infra')}]\n{val[:18]}"
        elif ntype == "capability":
            node_colors.append("#9b59b6")  # Purple for Attack Capabilities
            node_sizes.append(600)
            labels[node] = f"[CAP]\n{node_data.get('subtype', '')}"

    # 3. Compute layout coordinates
    plt.figure(figsize=(12, 8), dpi=120)
    pos = nx.spring_layout(subgraph, k=0.6, seed=42)

    # 4. Draw graph elements
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors="white",
        linewidths=1.5
    )
    nx.draw_networkx_edges(subgraph, pos, edge_color="#bdc3c7", alpha=0.6, width=1.2)
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8, font_family="sans-serif", font_weight="bold")

    plt.title(f"Forensic Stigmergy Cluster around {center_node} (Radius = {radius} Hops)", fontsize=13, pad=15)
    plt.axis("off")
    plt.tight_layout()

    # 5. Export or Render
    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)  # Create assets/ directory if missing
        plt.savefig(out_file, bbox_inches="tight")
        print(f"[+] Forensic graph successfully saved to: {out_file.resolve()}")
    else:
        plt.show()

    plt.close()  # Clean up memory buffer

import math
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from matplotlib import patheffects as pe
from main import findEgoIds, parseEdges

FOLDER = "data/twitter"
EGO = sys.argv[1] if len(sys.argv) > 1 else "87798068"
RAMP = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
INK, MUTED, EDGE = "#0b0b0b", "#52514e", "#c9c8c3"


def main():
    G = nx.DiGraph()
    for egoId in findEgoIds(FOLDER):
        edges = parseEdges(os.path.join(FOLDER, egoId + ".edges"))
        G.add_node(egoId)
        G.add_edges_from(edges)
        for n in {x for e in edges for x in e}:
            G.add_edge(egoId, n)
    U = G.to_undirected()
    U.remove_edges_from(nx.selfloop_edges(U))

    alters = {x for e in parseEdges(os.path.join(FOLDER, EGO + ".edges")) for x in e}
    H = U.subgraph(alters)
    followers = dict(G.in_degree(alters))
    clustering = nx.clustering(G, alters)
    pos = nx.spring_layout(H, seed=2222, k=0.35, iterations=200)
    fmax = max(followers.values())
    nodes = sorted(H.nodes(), key=lambda v: followers[v])
    sizes = [12 + 420 * math.sqrt(followers[v] / fmax) for v in nodes]
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color=EDGE, width=0.3, alpha=0.6)
    sc = ax.scatter([pos[v][0] for v in nodes], [pos[v][1] for v in nodes], s=sizes, c=[clustering[v] for v in nodes], cmap=RAMP, vmin=0, vmax=1, edgecolors="#fcfcfb", linewidths=0.6, zorder=3)
    for v in sorted(nodes, key=lambda v: -followers[v])[:3]:
        ax.annotate(f"{v}\n{followers[v]} followers", pos[v], xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7, color=INK, zorder=4, path_effects=[pe.withStroke(linewidth=3, foreground="#fcfcfb")])
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("Local clustering coefficient", color=MUTED)
    cb.outline.set_visible(False)
    for f in (10, 100, 1000):
        if f <= fmax:
            ax.scatter([], [], s=12 + 420 * math.sqrt(f / fmax), color="#9ec5f4", label=f"{f} followers")
    ax.legend(frameon=False, loc="lower left", fontsize=7, labelcolor=MUTED, scatterpoints=1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("report_figures/23_ego_network.pdf", bbox_inches="tight")
    fig.savefig("report_figures/23_ego_network.png", dpi=220, bbox_inches="tight")

    big = sorted(H.nodes(), key=lambda v: -followers[v])[:10]
    small = [v for v in H if followers[v] <= 20]

if __name__ == "__main__":
    main()

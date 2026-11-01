import json
import os
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import patheffects as pe
from main import findEgoIds, parseEdges
from plot_community_topics import LABELS, GROUP_COLOURS

FOLDER = "data/twitter"
PER_COMMUNITY = 50
N_BRIDGES = 10
INK, MUTED, EDGE, BRIDGE = "#0b0b0b", "#52514e", "#d4d3ce", "#0b0b0b"
HALO = [pe.withStroke(linewidth=3, foreground="#fcfcfb")]

def main():
    U = nx.Graph()
    for egoId in findEgoIds(FOLDER):
        edges = parseEdges(os.path.join(FOLDER, egoId + ".edges"))
        U.add_edges_from(edges)
        U.add_edges_from((egoId, n) for n in {x for e in edges for x in e})
    U.remove_edges_from(nx.selfloop_edges(U))
    comm = json.load(open("commMetrics.json"))
    membership = {v: int(c) for v, c in comm["leiden"]["membership"].items()}
    bridges = [v for v, _ in comm["bridges"]["topBetweenness"][:N_BRIDGES]]
    members = defaultdict(list)
    for v in U:
        c = membership[v]
        if c in LABELS:
            inside = sum(1 for u in U[v] if membership[u] == c)
            members[c].append((inside, v))
    core = {v for c in LABELS for _, v in sorted(members[c], reverse=True)[:PER_COMMUNITY]}
    nodes = core | set(bridges)
    H = U.subgraph(nodes)
    H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
    core = core & set(H)
    C = nx.Graph()
    C.add_nodes_from(LABELS)
    for p in comm["bridges"]["communityBridgeMetrics"]["communityPairs"]:
        a, b = p["communityA"], p["communityB"]
        if a in LABELS and b in LABELS:
            C.add_edge(a, b, dist=1 / (1 + p["numEdges"]) ** 0.25)
    centre = nx.kamada_kawai_layout(C, weight="dist", scale=1.0)
    pos = {}
    for c in LABELS:
        vs = [v for v in core if membership[v] == c and v not in bridges]
        local = nx.spring_layout(H.subgraph(vs), seed=2222, iterations=200)
        xs = sorted(x for x, _ in local.values()); ys = sorted(y for _, y in local.values())
        mx, my = xs[len(xs) // 2], ys[len(ys) // 2]
        radii = sorted(((x - mx) ** 2 + (y - my) ** 2) ** 0.5 for x, y in local.values())
        r90 = radii[int(0.9 * (len(radii) - 1))] or 1.0
        for v, (x, y) in local.items():
            dx, dy = (x - mx) / r90, (y - my) / r90
            d = (dx * dx + dy * dy) ** 0.5
            if d > 1.2:
                dx, dy = dx * 1.2 / d, dy * 1.2 / d
            pos[v] = (centre[c][0] + 0.12 * dx, centre[c][1] + 0.12 * dy)
    for v in bridges:
        if v in H:
            nb = [pos[u] for u in H[v] if u in pos]
            pos[v] = (sum(x for x, _ in nb) / len(nb), sum(y for _, y in nb) / len(nb))
    bridges = [v for v in bridges if v in H]
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color=EDGE, width=0.2, alpha=0.35)
    for c in LABELS:
        vs = [v for v in core if membership[v] == c and v not in bridges]
        label, group = LABELS[c]
        ax.scatter([pos[v][0] for v in vs], [pos[v][1] for v in vs], s=14, color=GROUP_COLOURS[group], edgecolors=MUTED if group is None else "#fcfcfb", linewidths=0.5 if group is None else 0.3, zorder=3)
        ax.annotate(label, (centre[c][0], centre[c][1] - 0.17), ha="center", va="top", fontsize=7, color=INK, zorder=5, path_effects=HALO)
    ax.scatter([pos[v][0] for v in bridges], [pos[v][1] for v in bridges], s=55, marker="D", color=BRIDGE, edgecolors="#fcfcfb", linewidths=1, zorder=4)
    for v in bridges[:3] + ["40981798"]:
        ax.annotate(v, pos[v], xytext=(5, 4), textcoords="offset points", fontsize=6.5, color=INK, zorder=5, path_effects=HALO)
    handles = [plt.scatter([], [], s=25, color=GROUP_COLOURS[g], edgecolors=MUTED if g is None else "none", linewidths=0.8, label=g or "No clear topic") for g in ["Gaming", "News & politics", "Tech & design", "Other topic", None]]
    handles.append(plt.scatter([], [], s=30, marker="D", color=BRIDGE, label="Top-10 betweenness"))
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=7, labelcolor=MUTED, bbox_to_anchor=(0, 1.02))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("report_figures/24_community_nodes.pdf", bbox_inches="tight")
    fig.savefig("report_figures/24_community_nodes.png", dpi=220, bbox_inches="tight")
    cross = sum(1 for a, b in H.edges() if membership[a] != membership[b])

if __name__ == "__main__":
    main()

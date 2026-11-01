import argparse
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import patheffects as pe

LABELS = {
    5: ("No clear topic", None),
    18: ("Console gaming", "Gaming"),
    0: ("Paramore fans", "Other topic"),
    7: ("Tech & startups", "Tech & design"),
    17: ("Call of Duty", "Gaming"),
    45: ("Space (NASA)", "Other topic"),
    30: ("Esports (SC2)", "Gaming"),
    62: ("US government", "News & politics"),
    120: ("Web design", "Tech & design"),
    6: ("Webcomics", "Other topic"),
    35: ("No clear topic", None),
    85: ("Food & chefs", "Other topic"),
    4: ("UK news (BBC)", "News & politics"),
    46: ("NFL", "Other topic"),
    29: ("Guild Wars 2", "Gaming"),
}
GROUP_COLOURS = {
    "Gaming": "#2a78d6",
    "News & politics": "#eb6834",
    "Tech & design": "#1baf7a",
    "Other topic": "#9a9893",
    None: "#ffffff",
}
INK, MUTED, EDGE = "#0b0b0b", "#52514e", "#b9b8b3"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comm", default="commMetrics.json")
    ap.add_argument("--out", default="report_figures/19_community_topics.pdf")
    ap.add_argument("--edges-per-node", type=int, default=3,
                    help="keep each community's strongest links")
    args = ap.parse_args()
    comm = json.load(open(args.comm))
    pairs = comm["bridges"]["communityBridgeMetrics"]["communityPairs"]
    sizes = {}
    for p in pairs:
        sizes[p["communityA"]] = p["sizeA"]
        sizes[p["communityB"]] = p["sizeB"]
    keep = set(LABELS)
    strongest = {c: [] for c in keep}
    for p in pairs:
        a, b = p["communityA"], p["communityB"]
        if a in keep and b in keep:
            strongest[a].append(p)
            strongest[b].append(p)
    chosen = {}
    for c in keep:
        for p in strongest[c][: args.edges_per_node]:
            chosen[(p["communityA"], p["communityB"])] = p["numEdges"]
    G = nx.Graph()
    for c in keep:
        G.add_node(c, size=sizes[c])
    for (a, b), w in chosen.items():
        G.add_edge(a, b, weight=w)
    for a, b, d in G.edges(data=True):
        d["dist"] = 1 + 2 * (1 - math.sqrt(d["weight"] / max(chosen.values())))
    pos = nx.kamada_kawai_layout(G, weight="dist")
    wmax = max(chosen.values())
    smax = max(sizes[c] for c in keep)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for a, b, d in G.edges(data=True):
        ax.plot(*zip(pos[a], pos[b]), color=EDGE, lw=0.6 + 3.4 * math.sqrt(d["weight"] / wmax), solid_capstyle="round", zorder=1)
    for c in G.nodes():
        label, group = LABELS[c]
        area = 120 + 1500 * sizes[c] / smax
        ax.scatter(*pos[c], s=area, color=GROUP_COLOURS[group], edgecolors="#fcfcfb" if group else MUTED, linewidths=2 if group else 0.8, zorder=3)
        r = math.sqrt(area) / 2 + 3
        ax.annotate(f"{label}\n{sizes[c]:,} users", pos[c], xytext=(0, -r), textcoords="offset points", ha="center", va="top", fontsize=7, color=INK, zorder=4, linespacing=1.1, path_effects=[pe.withStroke(linewidth=3, foreground="#fcfcfb")])
    handles = [plt.scatter([], [], s=60, color=GROUP_COLOURS[g], edgecolors=MUTED if g is None else "none", linewidths=0.8, label=g or "No clear topic") for g in ["Gaming", "News & politics", "Tech & design", "Other topic", None]]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7, labelcolor=MUTED, bbox_to_anchor=(0, 1.02))
    ax.margins(0.12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.rsplit(".", 1)[0] + ".png", dpi=220, bbox_inches="tight")

if __name__ == "__main__":
    main()

import glob
import json
import os
from collections import Counter
import igraph as ig
import numpy as np

FOLDER = "data/twitter"

def load_graph(folder):
    egos = sorted(os.path.basename(f)[: -len(".edges")] for f in glob.glob(os.path.join(folder, "*.edges")))
    edges, nodes = set(), set(egos)
    for ego in egos:
        base = os.path.join(folder, ego)
        in_file = set()
        for line in open(base + ".edges"):
            parts = line.split()
            if len(parts) == 2:
                edges.add((parts[0], parts[1]))
                in_file.update(parts)
        edges.update((ego, n) for n in in_file)
        nodes |= in_file
        if os.path.exists(base + ".feat"):
            nodes.update(l.split()[0] for l in open(base + ".feat") if l.strip())
        if os.path.exists(base + ".circles"):
            for line in open(base + ".circles"):
                nodes.update(line.split()[1:])
    names = sorted(nodes)
    index = {v: i for i, v in enumerate(names)}
    D = ig.Graph(n=len(names), edges=[(index[a], index[b]) for a, b in edges], directed=True)
    return D, names, index, edges, set(egos)

def main():
    D, names, index, edges, egos = load_graph(FOLDER)
    U = D.as_undirected(mode="collapse").simplify()
    comm = json.load(open("commMetrics.json"))
    membership = comm["leiden"]["membership"]
    mem = np.array([membership[v] for v in names])
    result = {"numNodes": D.vcount(), "numDirectedEdges": D.ecount(), "numUndirectedEdges": U.ecount()}
    big = U.maximal_cliques(min=70)
    members = {i for q in big for i in q}
    largest = comm["cliques"]["largestClique"]
    k = len(largest)
    present = sum(1 for a in largest for b in largest if a != b and (a, b) in edges)
    result["cliques"] = {
        "maxCoreNumber": max(U.coreness()),
        "numCliquesOf70Plus": len(big),
        "distinctUsersInCliquesOf70Plus": len(members),
        "communitiesOfThoseUsers": {str(c): n for c, n in Counter(int(mem[i]) for i in members).items()},
        "largestCliqueSize": k,
        "largestCliqueFollowsPresent": present,
        "largestCliqueFollowsPossible": k * (k - 1),
        "largestCliqueFollowShare": present / (k * (k - 1)),
    }
    sizes = Counter(membership.values())
    real = sorted((s for s in sizes.values() if s > 1), reverse=True)
    isolated = [i for i in range(U.vcount()) if U.degree(i) == 0]
    el = np.array(U.get_edgelist())
    result["communities"] = {
        "numCommunities": len(sizes),
        "numCommunitiesWithMoreThanOneUser": len(real),
        "smallestSize": min(real),
        "largestSize": max(real),
        "numIsolatedUsers": len(isolated),
        "allIsolatedUsersAreSingletons": all(sizes[mem[i]] == 1 for i in isolated),
        "numCommunitiesWith1000Plus": sum(s >= 1000 for s in real),
        "shareOfUsersInTenLargest": sum(real[:10]) / U.vcount(),
        "shareOfEdgesInsideCommunity": float(np.mean(mem[el[:, 0]] == mem[el[:, 1]])),
    }
    din, dout = np.array(D.indegree()), np.array(D.outdegree())
    de = np.array(D.get_edgelist())
    s, t = de[:, 0], de[:, 1]
    result["degreeAssortativity"] = {
        "outIn": float(np.corrcoef(dout[s], din[t])[0, 1]),
        "inIn": float(np.corrcoef(din[s], din[t])[0, 1]),
        "outOut": float(np.corrcoef(dout[s], dout[t])[0, 1]),
        "inOut": float(np.corrcoef(din[s], dout[t])[0, 1]),
        "undirected": U.assortativity_degree(directed=False),
    }
    art = U.articulation_points()
    def participation(i):
        nb = U.neighbors(i)
        counts = Counter(mem[nb])
        return 1 - sum((c / len(nb)) ** 2 for c in counts.values()), len(counts), counts[mem[i]] / len(nb)
    pcs = [participation(i)[0] for i in range(U.vcount()) if U.degree(i) > 0]
    top_btw = [v for v, _ in comm["bridges"]["topBetweenness"][:20]]
    top_nbc = [v for v, _ in comm["bridges"]["topByNumNeighbourCommunities"][:20]]
    in_rank = {v: r + 1 for r, v in enumerate(sorted(names, key=lambda v: -din[index[v]]))}
    topAccounts = []
    for v, b in comm["bridges"]["topBetweenness"][:10]:
        pc, ncomm, own = participation(index[v])
        topAccounts.append({"account": v, "betweenness": b, "inDegreeRank": in_rank[v], "neighbourCommunities": ncomm, "ownCommunityShare": own, "participation": pc, "isEgo": v in egos})
    result["bridges"] = {
        "numArticulationPoints": len(art),
        "numArticulationPointsThatAreEgos": sum(names[i] in egos for i in art),
        "bridgeEdges": [{"users": [names[a], names[b]], "degrees": [U.degree(a), U.degree(b)]} for a, b in (U.es[e].tuple for e in U.bridges())],
        "medianParticipationCoefficient": float(np.median(pcs)),
        "top20BetweennessAlsoTop20NeighbourCommunities": len(set(top_btw) & set(top_nbc)),
        "top20BetweennessThatAreEgos": sum(v in egos for v in top_btw),
        "top10Betweenness": topAccounts,
    }
    with open("community_checks.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()

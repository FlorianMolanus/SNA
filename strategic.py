import json
import os
from collections import Counter
import networkx as nx
from main import findEgoIds, parseEdges

FOLDER = "data/twitter"
ACCOUNTS = ["115485051", "813286", "40981798", "15846407", "116485573"]

def rank_in(top_list, v):
    ids = [str(u) for u, _ in top_list]
    return ids.index(v) + 1 if v in ids else None

def main():
    m = json.load(open("metrics.json"))
    comm = json.load(open("commMetrics.json"))
    hits = json.load(open("hitsMetrics.json"))
    n = m["numNodes"]
    followers = {v: round(c * (n - 1)) for v, c in m["inDegreeCentrality"].items()}
    following = {v: round(c * (n - 1)) for v, c in m["outDegreeCentrality"].items()}
    follower_rank = {v: i + 1 for i, v in enumerate(sorted(followers, key=lambda v: -followers[v]))}
    membership = {v: int(c) for v, c in comm["leiden"]["membership"].items()}
    U = nx.Graph()
    for egoId in findEgoIds(FOLDER):
        edges = parseEdges(os.path.join(FOLDER, egoId + ".edges"))
        U.add_edges_from(edges)
        U.add_edges_from((egoId, x) for x in {y for e in edges for y in e})
    U.remove_edges_from(nx.selfloop_edges(U))
    rows = []
    for v in ACCOUNTS:
        nb = Counter(membership[u] for u in U[v] if u != v)
        total = sum(nb.values())
        row = {
            "account": v,
            "followers": followers[v], "followerRank": follower_rank[v], "following": following[v],
            "betweennessRank": rank_in(comm["bridges"]["topBetweenness"], v),
            "pageRankRank": rank_in(hits["pageRank"], v),
            "authorityRank": rank_in(hits["authorities"], v),
            "hubRank": rank_in(hits["hubs"], v),
            "neighbourCommunities": len(nb),
            "ownCommunityShare": round(nb[membership[v]] / total, 2),
        }
        rows.append(row)
    json.dump(rows, open("strategic_nodes.json", "w"), indent=2)

if __name__ == "__main__":
    main()

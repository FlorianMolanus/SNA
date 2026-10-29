from collections import Counter
import networkx as nx
import igraph as ig
import random
import json
import sys



# 1. Network statistics - metrics
def nodesAndEdges(G):
  numNodes = G.number_of_nodes()
  numEdges = G.number_of_edges()
  return numNodes, numEdges

def degreeDistribution(G):
  inDegree = [d for _, d in G.in_degree()]
  outDegree = [d for _, d in G.out_degree()]
  inDistribution = Counter(inDegree)
  outDistribution = Counter(outDegree)
  return inDistribution, outDistribution

def centralityIndices(G):
  degreeCentrality = nx.degree_centrality(G)
  inDegreeCentrality = nx.in_degree_centrality(G)
  outDegreeCentrality = nx.out_degree_centrality(G)
  return degreeCentrality, inDegreeCentrality, outDegreeCentrality

def clusteringCoefficient(G):
  averageClustering = nx.average_clustering(G)
  transitivity = nx.transitivity(G)
  return averageClustering, transitivity

def networkDiameter(G):
  # Takes too long
  largest = max(nx.strongly_connected_components(G), key=len)
  H = G.subgraph(largest)
  # diameter = nx.diameter(H)
  diameter = nx.approximation.diameter(H)
  return diameter

def density(G):
  density = nx.density(G)
  return density

def numberOfConnectedComponents(G):
  weak = list(nx.weakly_connected_components(G))
  strong = list(nx.strongly_connected_components(G))
  numWeak, numStrong = len(weak), len(strong)
  weakSizes = sorted((len(c) for c in weak), reverse=True)
  strongSizes = sorted((len(c) for c in strong), reverse=True)
  return numWeak, numStrong, weakSizes, strongSizes

# 2. Communities
def cliques(U, minSize=3, topN=20, maxCliques=None):
  print("turning into ig graph")
  U3 = nx.k_core(U, minSize - 1)
  g = ig.Graph.from_networkx(U)
  names = g.vs["_nx_name"]                       # igraph index -> original node ID

  sizeCounts = Counter()
  membership = Counter()
  largest = []
  total = 0
  for clique in g.maximal_cliques(min=minSize):
    total += 1
    sizeCounts[len(clique)] += 1
    if len(clique) > len(largest):
      largest = clique
    membership.update(names[i] for i in clique)
  return {
    "numMaximalCliques": total,
    "sizeDistribution": dict(sorted(sizeCounts.items())),
    "largestCliqueSize": len(largest),
    "largestClique": [names[i] for i in largest],
    "topNodesByCliqueCount": membership.most_common(topN),
  }

def homophily(G):
  return {"degreeAssortivity": nx.degree_assortativity_coefficient(G)}


# Pass communities to analyse bridges between communities rather than just bridges in general
def bridges(U: nx.Graph, communities=None, topN=100, betweennessSamples=80):
  nodes = U.number_of_nodes()
  betweenness = nx.betweenness_centrality(U, k=betweennessSamples, seed=2222)
  articulation = set(nx.articulation_points(U))
  bridgeEdges = list(nx.bridges(U))

  results = {
    "numArticulationPoints": len(articulation),
    "numBridgeEdges": len(bridgeEdges),
    "topBetweenness": sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:topN],
    "topArticulationPointsByBetweenness": sorted(articulation, key=lambda v: betweenness[v], reverse=True)[:topN],
  }

  if communities:
    commOf = {v: i for i, c in enumerate(communities) for v in c}
    participation = {}   # 0 = all neighbours in own community, near 1 = spread over many communities
    neighbourComms = {}
    for v in U:
      counts = Counter(commOf[u] for u in U[v] if u != v)
      d = sum(counts.values())
      if d == 0:
        continue
      participation[v] = 1 - sum((c / d) ** 2 for c in counts.values())
      neighbourComms[v] = len(counts)
    crossEdges = sum(1 for u, v in U.edges() if commOf[u] != commOf[v])
    results["crossCommunityEdges"] = crossEdges
    results["crossCommunityEdgeShare"] = crossEdges / max(U.number_of_edges(), 1)
    results["topParticipationCoefficient"] = sorted(participation.items(), key=lambda x: x[1], reverse=True)[:topN]
    results["topByNumNeighbourCommunities"] = sorted(neighbourComms.items(), key=lambda x: x[1], reverse=True)[:topN]

  return results


def leidenCommunities(U: nx.Graph, seed=2222):
  random.seed(seed)
  ig.set_random_number_generator(random)
  nodes = list(U.nodes())
  index = {v: i for i, v in enumerate(nodes)}
  g = ig.Graph(n=len(nodes), edges=[(index[u], index[v]) for u,v in U.edges()])
  part = g.community_leiden(objective_function="modularity", n_iterations=-1)
  return [{nodes[i] for i in c} for c in part]


def partitionSummary(U, communities):
  sizes = sorted((len(c) for c in communities), reverse=True)
  return {
    "numCommunities": len(communities),
    "modularity": nx.community.modularity(U, communities),
    "numSingletons": sum(1 for s in sizes if s == 1),
    "largestSizes": sizes[:50],
  }


def runCommunityAnalysis(G: nx.DiGraph, U: nx.Graph, outputPath="commMetrics.json"):
  # U = G.to_undirected()
  U.remove_edges_from(nx.selfloop_edges(U))
  results = {}
  

  print("Calculating cliques...")
  results["cliques"] = cliques(U, minSize=70)

  print("homophily...")
  results["homophily"] = homophily(G)
 
  partitions = {}
  print("leiden...")
  partitions["leiden"] = leidenCommunities(U)

  for name, comms in partitions.items():
    entry = partitionSummary(U, comms)
    entry["membership"] = {str(v): i for i, c in enumerate(comms) for v in c}
    results[name] = entry

  print("bridges...")
  results["bridges"] = bridges(U, partitions["leiden"])
 
  with open(outputPath, "w") as f:
    json.dump(results, f, indent=2, default=str)
  return results, partitions


# 3. HITS algorithm
def HITS(G, outputPath="hitsMetrics.json"):
  hubs, authorities = nx.hits(G, max_iter=500)
  topHubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:50]
  topAuthorities =sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:50]
  results = {
    "hubs": topHubs,
    "authorities": topAuthorities 
  }
  with open(outputPath, "w") as f:
    json.dump(results, f, indent=2, default=str)


# 4. Longitudinal analysis
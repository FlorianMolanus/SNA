from collections import Counter
import networkx as nx



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
  sizeCounts = Counter()
  membership = Counter()
  largest = []
  total = 0
  for clique in nx.find_cliques(U):
    total += 1
    k = len(clique)
    sizeCounts[k] += 1
    if k > len(largest):
      largest = clique
    if k >= minSize:
      membership.update(clique)
    if maxCliques and total >= maxCliques:
      break
  return {
    "numMaximalCliques": total,
    "truncated": bool(maxCliques and total >= maxCliques),
    "sizeDistribution": dict(sorted(sizeCounts.items())),
    "largestCliqueSize": len(largest),
    "largestClique": list(largest),
    "topNodesByCliqueCount": membership.most_common(topN),
  }

def homophily(G):
  return {"degreeAssortivity": nx.degree_assortativity_coefficient(G)}




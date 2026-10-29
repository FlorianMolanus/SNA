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
  largest = max(nx.strongly_connected_components(G), key=len)
  H = G.subgraph(largest)
  diameter = nx.diameter(H)
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
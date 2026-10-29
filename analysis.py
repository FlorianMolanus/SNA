from collections import Counter
import networkx as nx



# 1. Network statistics - metrics
def countNodesAndEdges(G):
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
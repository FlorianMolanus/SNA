import os
import glob
import sys
import networkx as nx
import pickle
import time
from analysis import *


# Find all ego ids
def findEgoIds(folder):
  edgeFiles = glob.glob(os.path.join(folder, "*.edges"))
  egoIds = sorted(
    os.path.basename(f)[: -len(".edges")] for f in edgeFiles
  )
  return egoIds


# Get all edges in a given .edges file
def parseEdges(path):
  edges = []
  with open(path, "r") as f:
    for lineNum, line in enumerate(f, 1):
      line = line.strip()
      if not line:
        continue
      parts = line.split()
      if len(parts) != 2:
        print(f"  [warn] {path}:{lineNum} malformed line: {line!r}")
        continue
      a, b = parts
      edges.append((a, b))
  return edges


# Get all circles in a given .circles file
def parseCircles(path):
  circles = {}
  if not os.path.exists(path):
    return circles
  with open(path, "r") as f:
    for line in f:
      line = line.rstrip("\n")
      if not line.strip():
        continue
      parts = line.split("\t")
      if len(parts) == 1:
        parts = line.split()
      name = parts[0]
      members = [p for p in parts[1:] if p != ""]
      circles[name] = members
  return circles

# Get all featnames in a featnames file
def parseFeatnames(path):
  names = []
  if not os.path.exists(path):
    return names
  with open(path, "r", encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      idxStr, name = line.split(" ", 1)
      names.append(name)
  return names

# Get all features in a feat file
def parseFeat(path, featnames):
  result = {}
  if not os.path.exists(path):
    return result
  with open(path, "r") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      parts = line.split()
      nodeId = parts[0]
      values = parts[1:]
      featDict = {}
      for name, val in zip(featnames, values):
        featDict[name] = int(val)
      result[nodeId] = featDict
  return result

# Get all features in the egofeat file
def parseEgofeat(path, featnames, egoId):
  if not os.path.exists(path):
    return {}
  with open(path, "r") as f:
    line = f.readline().strip()
  if not line:
    return {}
  values = line.split()
  featDict = {}
  for name, val in zip(featnames, values):
    featDict[name] = int(val)
  return {egoId: featDict}

# Construct the network by parsing all seperate ego networks and combining them. If network has already been constructed, load it.
def loadEgoNetworks(folder, read):
  if read:
    print("reading network from file")
    G: nx.DiGraph = pickle.load(open("digraph.pickle", "rb"))
    return G
  G = nx.DiGraph()

  egoIds = findEgoIds(folder)
  print(f"Found {len(egoIds)} ego networks in {folder}")

  for egoId in egoIds:
    base = os.path.join(folder, egoId)
    edges = parseEdges(base + ".edges")
    circles = parseCircles(base + ".circles")
    featnames = parseFeatnames(base + ".featnames")
    feat = parseFeat(base + ".feat", featnames)
    egofeat = parseEgofeat(base + ".egofeat", featnames, egoId)

    G.add_node(egoId)
    G.nodes[egoId]['isEgo'] = True

    G.add_edges_from(edges)

    nodesInFile = set()
    for a, b in edges:
      nodesInFile.add(a)
      nodesInFile.add(b)

    for n in nodesInFile:
      G.add_edge(egoId, n)

    for circleName, members in circles.items():
      for m in members:
        G.add_node(m)
        nodeCircles = G.nodes[m].setdefault('circles', {})
        nodeCircles.setdefault(egoId, []).append(circleName)

    for nodeId, featDict in feat.items():
      G.add_node(nodeId)
      nodeFeatures = G.nodes[nodeId].setdefault('features', {})
      nodeFeatures[egoId] = featDict

    for eId, featDict in egofeat.items():
      nodeFeatures = G.nodes[eId].setdefault('features', {})
      nodeFeatures[eId] = featDict

  print(f"Combined graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
  pickle.dump(G, open('digraph.pickle', "wb"))
  return G


if __name__ == "__main__":

  folder = "data/twitter"
  startTime = time.time()
  G = loadEgoNetworks(folder, read=True)
  print("Loaded dataset in %s seconds" % (time.time() - startTime))

  # egoIds = findEgoIds(folder)
  print(clusteringCoefficient(G))


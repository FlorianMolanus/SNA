### Dataset
The dataset is created from ego nodes. The first ego node is 12831, the second is 78813, etc. These ego nodes are the nodes the researchers sampled. Edges are people that follow each other: a b means a follows b. The ego node follows every account that occurs in their .edges file. Circles are the circles that they are a part of. The feat files define the hashtags that people have used.

### Loading the network
For community metrics, the network is turned into an undirected graph. Both the directed graph and the undirected graph are serialized to save time on loading the network. On first run, run `loadEgoNetworks()` with `read=False`. The function will write the networks. You can then run with `read=True` all subsesequent times to save some time (unfortunately it still takes a minute, at least for me)

### Basic metrics
`metrics.json` contains the basic metrics about the dataset. It includes a diameter, which is an approximation due to the size of the dataset.

### Community metrics
Community metrics are in `commMetrics.json`. Because the dataset is fairly large, some compromises have to be made. Based on core analysis, the maximum size for a clique is 97.
`commMetricsOld.json` was generated with minimum size of 80 for the cliques, and a betweenness sample count of 80 for the bridges. `commMetrics` uses 60 and 200.

### Hits
Hits is calculated and stored in `hitsMetrics.json`.

### Longitudinal analysis
This is unfortunately not possible for our dataset, as there is no time data recorded. It could be worth looking for an extenstin elsewhere to compensate, but it's only 5 points so for now I am fine with skipping this.

### Visualize the graphs
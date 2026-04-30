import networkx as nx
#import pickle

G = nx.DiGraph()

with open("top-categories/wiki-topcats-page-names.txt", 'r', encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        node_id = int(parts[0])
        if len(parts) == 1:
            G.add_node(node_id)
        else:
            label = parts[1]
            G.add_node(node_id, label=label)

with open("top-categories/wiki-topcats.txt", 'r', encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        u, v = map(int, line.split())
        G.add_edge(u,v)

print("Nodes: ", G.number_of_nodes())
print("Edges: ", G.number_of_edges())
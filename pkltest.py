import networkx as nx
import pickle

with open("graph.pkl", "rb") as f:
    G2 = pickle.load(f)

print("here", G2.number_of_nodes(), G2.number_of_edges())
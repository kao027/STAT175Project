import pickle
import lzma

with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

with lzma.open("graph.pkl.xz", "wb", preset=9) as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

print("pickle compress done done")
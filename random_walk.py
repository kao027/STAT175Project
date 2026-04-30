import pickle
import random
from collections import defaultdict
from collections.abc import Mapping

try:
    import networkx as nx
except ImportError:
    nx = None


def load_network(pckl_path):
    """
    Load a pickled network object.

    Supported:
    - networkx Graph / DiGraph
    - adjacency dict: {node: [neighbors]}
    - adjacency dict of dicts: {node: {neighbor: weight}}
    """
    with open(pckl_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def is_networkx_graph(obj):
    return nx is not None and isinstance(obj, (nx.Graph, nx.DiGraph))


def get_all_nodes(graph):
    if is_networkx_graph(graph):
        return list(graph.nodes)

    if isinstance(graph, Mapping):
        return list(graph.keys())

    raise TypeError(
        "Unsupported graph type. Expected a networkx graph or adjacency dict."
    )


def get_neighbors(graph, node):
    """
    Returns a list of neighbors.
    Works for:
    - networkx graph
    - adjacency dict: {node: [neighbors]}
    - adjacency dict of dicts: {node: {neighbor: weight}}
    """
    if is_networkx_graph(graph):
        return list(graph.neighbors(node))

    if isinstance(graph, Mapping):
        nbrs = graph.get(node, [])
        if isinstance(nbrs, Mapping):
            return list(nbrs.keys())
        return list(nbrs)

    raise TypeError(
        "Unsupported graph type. Expected a networkx graph or adjacency dict."
    )


def choose_next_node(graph, current):
    """
    Choose next node uniformly at random among neighbors.
    """
    neighbors = get_neighbors(graph, current)
    if not neighbors:
        return None
    return random.choice(neighbors)


def single_random_walk(graph, start_node, walk_length):
    """
    Run one random walk and return a list of visited nodes.
    Includes the start node at index 0.
    """
    path = [start_node]
    current = start_node

    for _ in range(walk_length):
        nxt = choose_next_node(graph, current)
        if nxt is None:
            break
        path.append(nxt)
        current = nxt

    return path


def many_random_walks(
    graph,
    num_walks=10000,
    walk_length=20,
    start_nodes=None,
    seed=42,
):
    """
    Run many random walks and collect distance statistics.

    For every visit to a node at step t from the start:
        visit_counts[node] += 1
        distance_sum[node] += t

    Returns a dict with:
    - avg_visit_distance[node]
    - visit_counts[node]
    - raw_distance_sum[node]
    """
    random.seed(seed)

    nodes = get_all_nodes(graph)
    if not nodes:
        raise ValueError("Graph has no nodes.")

    if start_nodes is None:
        start_nodes = nodes
    else:
        start_nodes = [n for n in start_nodes if n in nodes]
        if not start_nodes:
            raise ValueError("None of the provided start_nodes are in the graph.")

    visit_counts = defaultdict(int)
    distance_sum = defaultdict(float)
    start_counts = defaultdict(int)

    for _ in range(num_walks):
        start = random.choice(start_nodes)
        start_counts[start] += 1

        path = single_random_walk(graph, start, walk_length)

        for step_index, node in enumerate(path):
            visit_counts[node] += 1
            distance_sum[node] += step_index

    avg_visit_distance = {}
    for node in visit_counts:
        avg_visit_distance[node] = distance_sum[node] / visit_counts[node]

    return {
        "avg_visit_distance": avg_visit_distance,
        "visit_counts": dict(visit_counts),
        "raw_distance_sum": dict(distance_sum),
        "start_counts": dict(start_counts),
    }


def rank_pages_by_avg_distance(results, min_visits=10, top_k=20):
    """
    Rank pages by average random-walk visit distance.
    Filters out rarely visited nodes to avoid noisy rankings.
    """
    avg_visit_distance = results["avg_visit_distance"]
    visit_counts = results["visit_counts"]

    rows = []
    for node, avg_dist in avg_visit_distance.items():
        count = visit_counts.get(node, 0)
        if count >= min_visits:
            rows.append((node, avg_dist, count))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]


def average_shortest_path_distance_per_node(graph):
    """
    Optional structural metric:
    For each node, compute average shortest-path distance
    from that node to all reachable nodes.

    Returns:
    {node: avg_shortest_path_distance}
    """
    if not is_networkx_graph(graph):
        raise TypeError(
            "Shortest-path calculation in this helper requires a networkx graph."
        )

    results = {}
    for node in graph.nodes:
        lengths = nx.single_source_shortest_path_length(graph, node)

        # Exclude self-distance 0
        distances = [d for target, d in lengths.items() if target != node]

        if distances:
            results[node] = sum(distances) / len(distances)
        else:
            results[node] = float("nan")

    return results


def print_top_ranked_pages(ranked_rows, title):
    print("\n" + title)
    print("-" * len(title))
    for i, (node, avg_dist, count) in enumerate(ranked_rows, start=1):
        print(f"{i:>2}. node={node!r} | avg_distance={avg_dist:.4f} | visits={count}")


if __name__ == "__main__":
    # =========================
    # SETTINGS
    # =========================
    pckl_file = "graph.pkl"
    num_walks = 50000
    walk_length = 30
    min_visits = 25
    top_k = 25
    seed = 42

    # Optional: limit random walk starting points
    # Example: start_nodes = ["PageA", "PageB"]
    start_nodes = None

    # =========================
    # LOAD GRAPH
    # =========================
    graph = load_network(pckl_file)
    print(f"Loaded graph object of type: {type(graph)}")

    # =========================
    # RANDOM-WALK DISTANCE ANALYSIS
    # =========================
    results = many_random_walks(
        graph,
        num_walks=num_walks,
        walk_length=walk_length,
        start_nodes=start_nodes,
        seed=seed,
    )

    ranked = rank_pages_by_avg_distance(
        results,
        min_visits=min_visits,
        top_k=top_k,
    )

    print_top_ranked_pages(
        ranked,
        title="Top pages by average random-walk distance",
    )

    # =========================
    # OPTIONAL: SHORTEST-PATH DISTANCE ANALYSIS
    # =========================
    if is_networkx_graph(graph):
        try:
            sp_avg = average_shortest_path_distance_per_node(graph)

            sp_rows = [
                (node, avg_dist)
                for node, avg_dist in sp_avg.items()
                if avg_dist == avg_dist  # filters NaN
            ]
            sp_rows.sort(key=lambda x: x[1], reverse=True)

            print("\nTop pages by average shortest-path distance")
            print("-------------------------------------------")
            for i, (node, avg_dist) in enumerate(sp_rows[:top_k], start=1):
                print(f"{i:>2}. node={node!r} | avg_shortest_path_distance={avg_dist:.4f}")

        except Exception as e:
            print("\nCould not compute shortest-path averages:")
            print(e)

    # =========================
    # OPTIONAL: SAVE RESULTS
    # =========================
    output_file = "random_walk_distance_results.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(
            {
                "settings": {
                    "num_walks": num_walks,
                    "walk_length": walk_length,
                    "min_visits": min_visits,
                    "top_k": top_k,
                    "seed": seed,
                    "start_nodes": start_nodes,
                },
                "random_walk_results": results,
                "ranked_random_walk_pages": ranked,
            },
            f,
        )

    print(f"\nSaved results to {output_file}")
import argparse
import csv
import math
import pickle
import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import mean, median

try:
    import networkx as nx
except ImportError:
    nx = None



# 
# Run with: 
# 
# python random_walk_to_categories.py \
#   --graph graph.pkl \
#   --pagerank pagerank.csv \
#   --categories top-categories/wiki-topcats-categories.txt \
#   --num-walks 100000 \
#   --max-steps 50 \
#   --top-seeds 100 \
#   --top-category-count 250 \
#   --output-prefix rw
#


# ============================================================
# User-adjustable defaults
# ============================================================
DEFAULT_TOP_CATEGORY_COUNT = 250


# ============================================================
# Graph helpers
# ============================================================

def load_network(pckl_path):
    with open(pckl_path, "rb") as f:
        return pickle.load(f)


def is_networkx_graph(obj):
    return nx is not None and isinstance(obj, (nx.Graph, nx.DiGraph))


def get_all_nodes(graph):
    if is_networkx_graph(graph):
        return list(graph.nodes)
    if isinstance(graph, Mapping):
        return list(graph.keys())
    raise TypeError("Unsupported graph type. Expected a networkx graph or adjacency dict.")


def get_neighbors(graph, node):
    if is_networkx_graph(graph):
        return list(graph.neighbors(node))
    if isinstance(graph, Mapping):
        nbrs = graph.get(node, [])
        if isinstance(nbrs, Mapping):
            return list(nbrs.keys())
        return list(nbrs)
    raise TypeError("Unsupported graph type. Expected a networkx graph or adjacency dict.")


def choose_next_node(graph, current, rng):
    neighbors = get_neighbors(graph, current)
    if not neighbors:
        return None
    return rng.choice(neighbors)


# ============================================================
# Input parsing
# ============================================================

def coerce_node_id(value):
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if not text:
        return text
    try:
        return int(text)
    except ValueError:
        return text


def parse_categories(categories_path):
    """
    Expected format per line:
        Category:People_from_Worcester; 1056 1057 1058

    Returns:
        node_to_categories: dict[node] -> set(categories)
        category_to_nodes: dict[category] -> set(nodes)
    """
    node_to_categories = defaultdict(set)
    category_to_nodes = defaultdict(set)

    with open(categories_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if ";" not in line:
                raise ValueError(f"Malformed category line {line_no}: missing ';'")

            category_part, nodes_part = line.split(";", 1)
            category = category_part.strip()
            if not category:
                raise ValueError(f"Malformed category line {line_no}: missing category name")

            node_tokens = nodes_part.strip().split()
            for token in node_tokens:
                node = coerce_node_id(token)
                node_to_categories[node].add(category)
                category_to_nodes[category].add(node)

    return dict(node_to_categories), dict(category_to_nodes)


def filter_to_largest_categories(node_to_categories, category_to_nodes, top_category_count):
    """
    Keep only the categories with the most assigned pages.

    If top_category_count is None, keep all categories.
    """
    if top_category_count is None:
        kept_categories = set(category_to_nodes.keys())
    else:
        ranked_categories = sorted(
            category_to_nodes.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        kept_categories = {category for category, _ in ranked_categories[:top_category_count]}

    filtered_category_to_nodes = {
        category: set(nodes)
        for category, nodes in category_to_nodes.items()
        if category in kept_categories
    }

    filtered_node_to_categories = {}
    for node, categories in node_to_categories.items():
        kept_for_node = set(categories) & kept_categories
        if kept_for_node:
            filtered_node_to_categories[node] = kept_for_node

    return filtered_node_to_categories, filtered_category_to_nodes


def detect_pagerank_columns(fieldnames):
    lowered = {name.lower().strip(): name for name in fieldnames}

    node_candidates = ["node_id", "node", "page", "page_id", "id", "vertex", "title"]
    score_candidates = ["pagerank", "page_rank", "score", "rank", "pr"]

    node_col = next((lowered[c] for c in node_candidates if c in lowered), None)
    score_col = next((lowered[c] for c in score_candidates if c in lowered), None)

    if node_col is None or score_col is None:
        raise ValueError(
            "Could not detect pagerank.csv columns. "
            f"Found columns: {fieldnames}. "
            "Need one node column (e.g. node_id, node, page_id, id) and one score column "
            "(e.g. pagerank, score, rank)."
        )

    return node_col, score_col


def load_top_pagerank_nodes(pagerank_csv_path, graph_nodes, top_n=None, min_score=None):
    graph_node_set = set(graph_nodes)
    rows = []

    with open(pagerank_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("pagerank.csv appears to have no header row.")
        node_col, score_col = detect_pagerank_columns(reader.fieldnames)

        for row in reader:
            node = coerce_node_id(row[node_col])
            if node not in graph_node_set:
                continue

            try:
                score = float(row[score_col])
            except (TypeError, ValueError):
                continue

            if min_score is not None and score < min_score:
                continue

            rows.append((node, score))

    rows.sort(key=lambda x: x[1], reverse=True)
    if top_n is not None:
        rows = rows[:top_n]

    if not rows:
        raise ValueError("No PageRank seed nodes from pagerank.csv matched nodes in the graph.")

    return rows


# ============================================================
# Random-walk analysis
# ============================================================

def single_walk_hits_excluding_start(graph, start_node, node_to_categories, max_steps, rng):
    """
    Perform one random walk and record the first step index at which each category is hit,
    EXCLUDING the start node itself.

    Returns:
        seed_categories: set of categories already present at the start node
        first_hit_after_move: dict[category] -> first step reached, where the first possible
                              step is 1
    """
    seed_categories = set(node_to_categories.get(start_node, set()))
    first_hit_after_move = {}
    current = start_node

    for step in range(1, max_steps + 1):
        nxt = choose_next_node(graph, current, rng)
        if nxt is None:
            break
        current = nxt

        categories = node_to_categories.get(current)
        if categories:
            for category in categories:
                if category not in first_hit_after_move:
                    first_hit_after_move[category] = step

    return seed_categories, first_hit_after_move


def many_walks_category_hitting_times(
    graph,
    start_nodes,
    node_to_categories,
    category_to_nodes,
    num_walks,
    max_steps,
    seed=42,
):
    rng = random.Random(seed)

    if not start_nodes:
        raise ValueError("start_nodes is empty.")

    start_counts = defaultdict(int)
    hit_steps_after_move = defaultdict(list)
    seed_hits = defaultdict(int)
    misses_after_move = defaultdict(int)

    all_categories = set(category_to_nodes.keys())

    for _ in range(num_walks):
        start = rng.choice(start_nodes)
        start_counts[start] += 1

        seed_categories, first_hit_after_move = single_walk_hits_excluding_start(
            graph=graph,
            start_node=start,
            node_to_categories=node_to_categories,
            max_steps=max_steps,
            rng=rng,
        )

        for category in seed_categories:
            seed_hits[category] += 1

        for category, step in first_hit_after_move.items():
            hit_steps_after_move[category].append(step)

        missed_this_walk = all_categories - set(first_hit_after_move.keys())
        for category in missed_this_walk:
            misses_after_move[category] += 1

    summary_rows = []
    for category in all_categories:
        steps = hit_steps_after_move.get(category, [])
        hits_after_move = len(steps)
        seed_hit_count = seed_hits.get(category, 0)
        miss_count = misses_after_move.get(category, 0)
        hit_rate_after_move = hits_after_move / num_walks if num_walks else math.nan
        avg_hit_length = mean(steps) if steps else math.nan
        median_hit_length = median(steps) if steps else math.nan

        summary_rows.append(
            {
                "category": category,
                "category_size": len(category_to_nodes[category]),
                "seed_hits": seed_hit_count,
                "hits_after_move": hits_after_move,
                "misses_after_move": miss_count,
                "hit_rate_after_move": hit_rate_after_move,
                "avg_hit_length": avg_hit_length,
                "median_hit_length": median_hit_length,
            }
        )

    summary_rows.sort(
        key=lambda row: (
            math.inf if math.isnan(row["avg_hit_length"]) else row["avg_hit_length"],
            -row["hit_rate_after_move"],
            row["category"],
        )
    )

    return {
        "summary_rows": summary_rows,
        "hit_steps_after_move": {k: list(v) for k, v in hit_steps_after_move.items()},
        "seed_hits": dict(seed_hits),
        "misses_after_move": dict(misses_after_move),
        "start_counts": dict(start_counts),
    }


# ============================================================
# Output helpers
# ============================================================

def write_summary_csv(rows, output_path):
    fieldnames = [
        "category",
        "category_size",
        "seed_hits",
        "hits_after_move",
        "misses_after_move",
        "hit_rate_after_move",
        "avg_hit_length",
        "median_hit_length",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_seed_csv(seed_rows, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "pagerank"])
        writer.writerows(seed_rows)


def print_top_categories(rows, top_k=25):
    print("\nTop categories by smallest average first-hit walk length (excluding seed node)")
    print("-------------------------------------------------------------------------")
    for i, row in enumerate(rows[:top_k], start=1):
        avg_text = "NaN" if math.isnan(row["avg_hit_length"]) else f"{row['avg_hit_length']:.4f}"
        print(
            f"{i:>2}. {row['category']} | "
            f"category_size={row['category_size']} | "
            f"avg_hit_length={avg_text} | "
            f"hit_rate_after_move={row['hit_rate_after_move']:.4%} | "
            f"hits_after_move={row['hits_after_move']} | "
            f"seed_hits={row['seed_hits']} | "
            f"misses_after_move={row['misses_after_move']}"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run many random walks from high-PageRank seed pages and compute "
            "average first-hit walk length to the largest categories, excluding the start node."
        )
    )
    parser.add_argument("--graph", default="graph.pkl", help="Path to pickled graph")
    parser.add_argument("--pagerank", default="pagerank.csv", help="Path to pagerank CSV")
    parser.add_argument(
        "--categories",
        default="top-categories/wiki-topcats-categories.txt",
        help="Path to category assignment file",
    )
    parser.add_argument("--num-walks", type=int, default=100000, help="Number of random walks")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per walk")
    parser.add_argument(
        "--top-seeds",
        type=int,
        default=100,
        help="Use the top N PageRank pages as random-walk starting seeds",
    )
    parser.add_argument(
        "--top-category-count",
        type=int,
        default=DEFAULT_TOP_CATEGORY_COUNT,
        help="Keep only the N categories with the most pages. Set to 250 by default.",
    )
    parser.add_argument(
        "--min-pagerank",
        type=float,
        default=None,
        help="Optional minimum PageRank threshold for starting seeds",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-prefix",
        default="rw_category_results_top250",
        help="Prefix for output files",
    )
    args = parser.parse_args()

    graph = load_network(args.graph)
    graph_nodes = get_all_nodes(graph)
    print(f"Loaded graph: {type(graph)} with {len(graph_nodes):,} nodes")

    node_to_categories, category_to_nodes = parse_categories(args.categories)
    print(f"Loaded {len(category_to_nodes):,} total categories from {args.categories}")

    node_to_categories, category_to_nodes = filter_to_largest_categories(
        node_to_categories=node_to_categories,
        category_to_nodes=category_to_nodes,
        top_category_count=args.top_category_count,
    )
    print(f"Keeping {len(category_to_nodes):,} largest categories")

    seed_rows = load_top_pagerank_nodes(
        pagerank_csv_path=args.pagerank,
        graph_nodes=graph_nodes,
        top_n=args.top_seeds,
        min_score=args.min_pagerank,
    )
    start_nodes = [node for node, _ in seed_rows]
    print(f"Using {len(start_nodes):,} high-PageRank seed pages from {args.pagerank}")

    analysis = many_walks_category_hitting_times(
        graph=graph,
        start_nodes=start_nodes,
        node_to_categories=node_to_categories,
        category_to_nodes=category_to_nodes,
        num_walks=args.num_walks,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    summary_rows = analysis["summary_rows"]
    print_top_categories(summary_rows, top_k=25)

    output_prefix = Path(args.output_prefix)
    summary_csv = f"{output_prefix}_category_summary.csv"
    seed_csv = f"{output_prefix}_seed_pages.csv"
    pkl_path = f"{output_prefix}_full_results.pkl"

    write_summary_csv(summary_rows, summary_csv)
    write_seed_csv(seed_rows, seed_csv)

    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "settings": vars(args),
                "seed_rows": seed_rows,
                "num_graph_nodes": len(graph_nodes),
                "num_categories_after_filter": len(category_to_nodes),
                "analysis": analysis,
            },
            f,
        )

    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved seed CSV:    {seed_csv}")
    print(f"Saved full pickle: {pkl_path}")


if __name__ == "__main__":
    main()

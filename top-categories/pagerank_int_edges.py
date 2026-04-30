#!/usr/bin/env python3
"""
PageRank for a graph stored as lines of:

    TARGET SOURCE

Example:
    1 2957573
    1 57329
    2 59502

This means:
    2957573 -> 1
    57329   -> 1
    59502   -> 2

Usage:
    python pagerank_int_edges.py graph.txt --top-k 50 --output pagerank.csv

Requires:
    pip install numpy scipy
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix


def parse_graph_file(path: str):
    """
    Parse file with lines:
        TARGET SOURCE

    Returns:
        node_ids: sorted numpy array of unique node IDs
        edges: list of (src, dst)
        outdegree: dict[src] = number of outgoing edges
    """
    all_nodes = set()
    edges = []
    outdegree = defaultdict(int)

    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Line {line_num} malformed: {line!r}")

            try:
                dst = int(parts[0])   # target
                src = int(parts[1])   # source
            except ValueError as e:
                raise ValueError(f"Line {line_num} contains non-integer IDs: {line!r}") from e

            all_nodes.add(src)
            all_nodes.add(dst)
            edges.append((src, dst))
            outdegree[src] += 1

    node_ids = np.array(sorted(all_nodes), dtype=np.int64)
    return node_ids, edges, outdegree


def build_sparse_matrix(node_ids, edges, outdegree):
    """
    Build sparse transition matrix M where:

        rank_next = alpha * (M @ rank) + teleport + dangling_mass

    For each edge src -> dst:
        M[dst, src] = 1 / outdegree[src]
    """
    n = len(node_ids)
    id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    rows = np.empty(len(edges), dtype=np.int32)
    cols = np.empty(len(edges), dtype=np.int32)
    data = np.empty(len(edges), dtype=np.float64)

    for i, (src, dst) in enumerate(edges):
        src_idx = id_to_idx[src]
        dst_idx = id_to_idx[dst]
        rows[i] = dst_idx
        cols[i] = src_idx
        data[i] = 1.0 / outdegree[src]

    M = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    dangling = np.array(
        [outdegree.get(node_id, 0) == 0 for node_id in node_ids],
        dtype=np.float64,
    )

    return M, dangling


def pagerank(M, dangling, alpha=0.85, tol=1e-8, max_iter=100, verbose=True):
    """
    Power-iteration PageRank.
    """
    n = M.shape[0]
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = np.full(n, (1.0 - alpha) / n, dtype=np.float64)

    for it in range(1, max_iter + 1):
        prev = rank

        dangling_mass = prev[dangling == 1.0].sum()

        rank = alpha * (M @ prev)
        rank += alpha * dangling_mass / n
        rank += teleport

        # stabilize against tiny floating-point drift
        rank /= rank.sum()

        delta = np.abs(rank - prev).sum()
        if verbose:
            print(f"Iteration {it:3d}: L1 delta = {delta:.3e}")

        if delta < tol:
            return rank, it

    return rank, max_iter


def main():
    parser = argparse.ArgumentParser(description="Run PageRank on integer edge list.")
    parser.add_argument("input_file", help="Path to graph file")
    parser.add_argument("--alpha", type=float, default=0.85, help="Damping factor")
    parser.add_argument("--tol", type=float, default=1e-8, help="Convergence tolerance")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top nodes to print")
    parser.add_argument("--output", type=str, default="", help="Optional CSV output path")
    parser.add_argument("--quiet", action="store_true", help="Suppress iteration logs")
    args = parser.parse_args()

    t0 = time.time()
    print("Parsing graph...")
    node_ids, edges, outdegree = parse_graph_file(args.input_file)
    t1 = time.time()

    print(f"Nodes: {len(node_ids):,}")
    print(f"Edges: {len(edges):,}")
    print(f"Parse time: {t1 - t0:.2f}s")

    print("Building sparse matrix...")
    M, dangling = build_sparse_matrix(node_ids, edges, outdegree)
    t2 = time.time()
    print(f"Matrix build time: {t2 - t1:.2f}s")

    print("Running PageRank...")
    ranks, num_iter = pagerank(
        M,
        dangling,
        alpha=args.alpha,
        tol=args.tol,
        max_iter=args.max_iter,
        verbose=not args.quiet,
    )
    t3 = time.time()

    print(f"Converged in {num_iter} iterations")
    print(f"PageRank time: {t3 - t2:.2f}s")
    print(f"Total time: {t3 - t0:.2f}s")

    top_k = min(args.top_k, len(node_ids))
    top_idx = np.argsort(-ranks)[:top_k]

    print(f"\nTop {top_k} nodes by PageRank:")
    print(f"{'Rank':>4}  {'Node ID':>12}  {'Score':>14}")
    print("-" * 36)
    for i, idx in enumerate(top_idx, start=1):
        print(f"{i:>4}  {node_ids[idx]:>12}  {ranks[idx]:>14.10f}")

    if args.output:
        order = np.argsort(-ranks)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("node_id,pagerank\n")
            for idx in order:
                f.write(f"{int(node_ids[idx])},{ranks[idx]:.12f}\n")
        print(f"\nSaved full results to {args.output}")


if __name__ == "__main__":
    main()
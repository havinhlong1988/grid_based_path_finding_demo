#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A* path-finding algorithm.

Cost:
  edge distance * average slowness * label penalty

Blocked:
  RA nodes are removed before graph search.
"""

from __future__ import annotations

import heapq
import math
import time

from src.model_io import iter_neighbors, edge_cost, heuristic_cost


def run(model, graph, start_idx: int, end_idx: int) -> dict:
    t0 = time.time()

    if start_idx not in graph["valid_indices"]:
        return {
            "success": False,
            "message": "Start node is blocked.",
            "path_indices": [],
        }

    if end_idx not in graph["valid_indices"]:
        return {
            "success": False,
            "message": "End node is blocked.",
            "path_indices": [],
        }

    open_heap = []
    heapq.heappush(open_heap, (0.0, start_idx))

    came_from = {}

    g_score = {start_idx: 0.0}
    f_score = {
        start_idx: heuristic_cost(model, graph, start_idx, end_idx)
    }

    visited = set()
    expanded_nodes = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in visited:
            continue

        visited.add(current)
        expanded_nodes += 1

        if current == end_idx:
            path = reconstruct_path(came_from, current)
            total_cost = float(g_score[end_idx])

            return {
                "success": True,
                "algorithm": "astar",
                "message": "Path found.",
                "path_indices": path,
                "total_cost": total_cost,
                "expanded_nodes": expanded_nodes,
                "visited_nodes": len(visited),
                "runtime_seconds": time.time() - t0,
            }

        for neighbor in iter_neighbors(model, graph, current):
            tentative_g = g_score[current] + edge_cost(model, graph, current, neighbor)

            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic_cost(model, graph, neighbor, end_idx)
                f_score[neighbor] = f
                heapq.heappush(open_heap, (f, neighbor))

    return {
        "success": False,
        "algorithm": "astar",
        "message": "No path found.",
        "path_indices": [],
        "total_cost": None,
        "expanded_nodes": expanded_nodes,
        "visited_nodes": len(visited),
        "runtime_seconds": time.time() - t0,
    }


def reconstruct_path(came_from: dict, current: int) -> list[int]:
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path
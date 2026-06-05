#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flood-fill / BFS path finder.

This ignores slowness and finds the minimum number of grid steps.
Useful as a pure connectivity test:
  - Can the drone move from DB to DK without crossing RA?
"""

from __future__ import annotations

from collections import deque
import time

from src.model_io import iter_neighbors


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

    queue = deque([start_idx])
    came_from = {start_idx: None}
    visited = {start_idx}

    while queue:
        current = queue.popleft()

        if current == end_idx:
            path = reconstruct_path(came_from, end_idx)

            return {
                "success": True,
                "algorithm": "flood_fill",
                "message": "Path found.",
                "path_indices": path,
                "total_cost": len(path) - 1,
                "expanded_nodes": len(visited),
                "visited_nodes": len(visited),
                "runtime_seconds": time.time() - t0,
            }

        for neighbor in iter_neighbors(model, graph, current):
            if neighbor in visited:
                continue

            visited.add(neighbor)
            came_from[neighbor] = current
            queue.append(neighbor)

    return {
        "success": False,
        "algorithm": "flood_fill",
        "message": "No path found.",
        "path_indices": [],
        "total_cost": None,
        "expanded_nodes": len(visited),
        "visited_nodes": len(visited),
        "runtime_seconds": time.time() - t0,
    }


def reconstruct_path(came_from: dict, end_idx: int) -> list[int]:
    path = []
    current = end_idx

    while current is not None:
        path.append(current)
        current = came_from[current]

    path.reverse()
    return path
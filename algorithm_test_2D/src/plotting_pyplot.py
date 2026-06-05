#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot model and path report.

Output:
  output/figures/senario1/path_report_{algorithm}.png

The report includes:
  1. XY model map with path footprint
  2. Path elevation/depth profile if the model is 3D
  3. Path slowness profile
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_path_report(
    model: pd.DataFrame,
    path_indices: list[int],
    figure_file: Path,
    algorithm_name: str,
    max_model_points: int = 300000,
    dpi: int = 300,
    model_alpha: float = 0.45,
    model_marker_size: float = 2.0,
    path_line_width: float = 2.0,
):
    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)

    if len(path_indices) == 0:
        raise ValueError("Cannot plot empty path.")

    path_df = model.loc[path_indices].copy().reset_index(drop=False)
    path_df["path_step"] = np.arange(len(path_df))

    plot_model = sample_model_for_plot(model, max_model_points=max_model_points)

    is_3d = model["z"].nunique() > 1

    if is_3d:
        fig = plt.figure(figsize=(11, 10))
        ax_map = fig.add_axes([0.08, 0.39, 0.86, 0.55])
        ax_z = fig.add_axes([0.08, 0.22, 0.86, 0.11])
        ax_slow = fig.add_axes([0.08, 0.07, 0.86, 0.11])
    else:
        fig = plt.figure(figsize=(10, 8))
        ax_map = fig.add_axes([0.08, 0.22, 0.86, 0.70])
        ax_slow = fig.add_axes([0.08, 0.07, 0.86, 0.10])
        ax_z = None

    plot_model_xy(
        ax=ax_map,
        model=plot_model,
        path_df=path_df,
        algorithm_name=algorithm_name,
        model_alpha=model_alpha,
        model_marker_size=model_marker_size,
        path_line_width=path_line_width,
    )

    if is_3d:
        plot_path_z_profile(ax_z, path_df)

    plot_path_slowness_profile(ax_slow, path_df)

    fig.suptitle(
        f"Scenario 1 path report - {algorithm_name}",
        fontsize=14,
        y=0.98,
    )

    fig.savefig(figure_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return figure_file


def sample_model_for_plot(model: pd.DataFrame, max_model_points: int = 300000) -> pd.DataFrame:
    """
    Large grid models can be too heavy to plot directly.
    This keeps all special labels and samples normal nodes if needed.
    """
    n = len(model)

    if n <= max_model_points:
        return model.copy()

    special = model[model["label_prefix"].isin(["DB", "DK", "FLZ", "RA"])].copy()
    normal = model[~model.index.isin(special.index)].copy()

    remain = max(max_model_points - len(special), 1000)

    if len(normal) > remain:
        normal = normal.sample(n=remain, random_state=42)

    out = pd.concat([normal, special], axis=0)
    return out.sort_index()


def plot_model_xy(
    ax,
    model: pd.DataFrame,
    path_df: pd.DataFrame,
    algorithm_name: str,
    model_alpha: float = 0.45,
    model_marker_size: float = 2.0,
    path_line_width: float = 2.0,
):
    """
    Plot XY footprint of model and path.

    Label convention:
      N    = model node
      DB   = drone base
      DK   = docking
      FLZ  = fly-control zone
      RA   = restricted airspace
    """

    label_order = [
        ("N", "Nodes"),
        ("FLZ", "Fly-control zone"),
        ("RA", "Restricted airspace"),
        ("DB", "Drone base"),
        ("DK", "Docking"),
    ]

    for prefix, name in label_order:
        sub = model[model["label_prefix"] == prefix]

        if sub.empty:
            continue

        ax.scatter(
            sub["x"],
            sub["y"],
            s=model_marker_size if prefix == "N" else model_marker_size * 8,
            alpha=model_alpha if prefix == "N" else 0.85,
            label=f"{name} ({len(sub):,})",
        )

    ax.plot(
        path_df["x"],
        path_df["y"],
        linewidth=path_line_width,
        label=f"Path - {algorithm_name}",
        zorder=10,
    )

    ax.scatter(
        path_df.iloc[0]["x"],
        path_df.iloc[0]["y"],
        s=90,
        marker="*",
        label=f"Start: {path_df.iloc[0]['label']}",
        zorder=20,
    )

    ax.scatter(
        path_df.iloc[-1]["x"],
        path_df.iloc[-1]["y"],
        s=90,
        marker="X",
        label=f"End: {path_df.iloc[-1]['label']}",
        zorder=20,
    )

    ax.set_xlabel("X / Longitude")
    ax.set_ylabel("Y / Latitude")
    ax.set_title("Model nodes and path footprint")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)

    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass


def plot_path_z_profile(ax, path_df: pd.DataFrame):
    ax.plot(
        path_df["path_step"],
        path_df["z"],
        linewidth=1.5,
    )

    ax.scatter(
        path_df["path_step"],
        path_df["z"],
        s=8,
    )

    ax.set_xlabel("Path step")
    ax.set_ylabel("Z / Altitude")
    ax.set_title("Path altitude profile")
    ax.grid(True, linewidth=0.4, alpha=0.4)


def plot_path_slowness_profile(ax, path_df: pd.DataFrame):
    ax.plot(
        path_df["path_step"],
        path_df["slowness"],
        linewidth=1.5,
    )

    ax.scatter(
        path_df["path_step"],
        path_df["slowness"],
        s=8,
    )

    ax.set_xlabel("Path step")
    ax.set_ylabel("Slowness")
    ax.set_title("Path slowness profile")
    ax.grid(True, linewidth=0.4, alpha=0.4)
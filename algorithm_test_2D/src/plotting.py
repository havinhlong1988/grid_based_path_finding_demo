#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pygmt


# ============================================================
# Main report plot
# ============================================================

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
    plot_model_as_flyable_nofly: bool = True,
    plot_no_fly_prefixes=("RA",),
    plot_no_fly_slowness_threshold: float = 1e5,
    plot_show_flz_overlay: bool = True,
    always_flyable_prefixes=("DB", "DK"),
    result: dict | None = None,
):
    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)

    if len(path_indices) == 0:
        raise ValueError("Cannot plot empty path.")

    path_df = model.loc[path_indices].copy().reset_index(drop=False)

    if "node_index" in path_df.columns:
        path_df = path_df.rename(columns={"node_index": "original_node_index"})
    elif "index" in path_df.columns:
        path_df = path_df.rename(columns={"index": "original_node_index"})

    path_df["path_step"] = np.arange(len(path_df))
    
    path_df = add_path_traveltime_columns(
        model=model,
        path_df=path_df,
        path_indices=path_indices,
    )

    plot_model = sample_model_for_plot(
        model,
        max_model_points=max_model_points,
    )

    region = get_xy_region_with_padding(plot_model, path_df, padding_ratio=0.04)
    is_lonlat = detect_lonlat(plot_model)

    fig = pygmt.Figure()

    pygmt.config(
        FONT_TITLE="13p,Helvetica-Bold",
        FONT_LABEL="10p,Helvetica",
        FONT_ANNOT_PRIMARY="8p,Helvetica",
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xxx",
    )

    projection = "M16c"
    title = f"Scenario 1 path report - {algorithm_name}"

    fig.basemap(
        region=region,
        projection=projection,
        frame=[
            "WSne+t" + title,
            "xaf+lLongitude" if is_lonlat else "xaf+lX",
            "yaf+lLatitude" if is_lonlat else "yaf+lY",
        ],
    )

    # Plot binary flyable / no-fly map
    plot_model_flyable_nofly(
        fig=fig,
        model=plot_model,
        model_marker_size=model_marker_size,
        model_alpha=model_alpha,
        no_fly_prefixes=plot_no_fly_prefixes,
        no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
        show_flz_overlay=plot_show_flz_overlay,
        always_flyable_prefixes=always_flyable_prefixes,
    )

    # Path
    fig.plot(
        x=path_df["x"],
        y=path_df["y"],
        pen=f"{path_line_width}p,black,--",
        label=f"Path - {algorithm_name}",
    )

    # Start / end
    start = path_df.iloc[0]
    end = path_df.iloc[-1]

    fig.plot(
        x=[start["x"]],
        y=[start["y"]],
        style="a0.45c",
        fill="yellow",
        pen="0.7p,black",
        label=f"Start: {start['label']}",
        transparency=50,
    )

    fig.plot(
        x=[end["x"]],
        y=[end["y"]],
        fill="grey@50",
        style="s0.42c",
        pen="1.4p,blue",
        label=f"End: {end['label']}",
        transparency=50,
    )
    
    # # Text report box at top-right
    # add_report_text_box(
    #     fig=fig,
    #     region=region,
    #     algorithm_name=algorithm_name,
    #     path_df=path_df,
    #     result=result,
    # )

    if is_lonlat:
        try:
            fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l1 km")
        except Exception:
            pass
            
    fig.legend(
        position="JTL+jTL+o0.15c/0.15c",
        box="+gwhite@25+p0.5p,black",
    )

    # ========================================================
    # Path slowness profile
    # ========================================================
    fig.shift_origin(xshift="16.5c",yshift="7.4c")

    slow_region = get_profile_region(
        x=path_df["path_step"].values,
        y=path_df["slowness"].values,
        y_padding_ratio=0.12,
    )

    fig.basemap(
        region=slow_region,
        projection="X10c/4.0c",
        frame=[
            "wSnE+tPath slowness profile",
            "xaf+lPath step",
            "yaf+lSlowness",
        ],
    )

    fig.plot(
        x=path_df["path_step"],
        y=path_df["slowness"],
        pen="1.2p,black,--",
    )

    fig.plot(
        x=path_df["path_step"],
        y=path_df["slowness"],
        style="c0.15c",
        fill="red",
        pen="0.1p,black",
    )

    # Report text box at right-top of slowness profile
    add_report_text_box_to_profile(
        fig=fig,
        region=slow_region,
        algorithm_name=algorithm_name,
        path_df=path_df,
        result=result,
    )

    # ========================================================
    # Cumulative traveltime profile
    # ========================================================
    fig.shift_origin(yshift="-7.4c")

    time_region = get_profile_region(
        x=path_df["path_step"].values,
        y=path_df["cumulative_traveltime_s"].values,
        y_padding_ratio=0.12,
    )

    fig.basemap(
        region=time_region,
        projection="X10c/4.0c",
        frame=[
            "wSnE+tCumulative traveltime profile",
            "xaf+lPath step",
            "yaf+lTraveltime (s)",
        ],
    )

    fig.plot(
        x=path_df["path_step"],
        y=path_df["cumulative_traveltime_s"],
        pen="1.2p,black,--",
    )

    fig.plot(
        x=path_df["path_step"],
        y=path_df["cumulative_traveltime_s"],
        style="c0.15c",
        fill="blue",
        pen="0.1p,black",
    )

    # Altitude profile for 3D
    if model["z"].nunique() > 1:
        fig.shift_origin(yshift="-4.2c")

        z_region = get_profile_region(
            x=path_df["path_step"].values,
            y=path_df["z"].values,
            y_padding_ratio=0.12,
        )

        fig.basemap(
            region=z_region,
            projection="X16c/3.0c",
            frame=[
                "WSne+tPath altitude profile",
                "xaf+lPath step",
                "yaf+lZ / altitude",
            ],
        )

        fig.plot(
            x=path_df["path_step"],
            y=path_df["z"],
            pen="1.2p,black",
        )

        fig.plot(
            x=path_df["path_step"],
            y=path_df["z"],
            style="c0.05c",
            fill="black",
            pen="0.1p,black",
        )

    fig.savefig(str(figure_file), dpi=dpi)
    return figure_file


# ============================================================
# Initiate plot
# ============================================================

def plot_initiate_model(
    model: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    figure_file: Path,
    max_model_points: int = 300000,
    dpi: int = 300,
    model_alpha: float = 0.45,
    model_marker_size: float = 2.0,
    plot_model_as_flyable_nofly: bool = True,
    plot_no_fly_prefixes=("RA",),
    plot_no_fly_slowness_threshold: float = 1e5,
    plot_show_flz_overlay: bool = True,
    always_flyable_prefixes=("DB", "DK"),
):
    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)

    plot_model = sample_model_for_plot(
        model,
        max_model_points=max_model_points,
    )

    start = model.loc[start_idx]
    end = model.loc[end_idx]

    point_df = pd.DataFrame(
        {
            "x": [start["x"], end["x"]],
            "y": [start["y"], end["y"]],
            "z": [start["z"], end["z"]],
            "label": [start["label"], end["label"]],
        }
    )

    region = get_xy_region_with_padding(
        model=plot_model,
        path_df=point_df,
        padding_ratio=0.04,
    )

    is_lonlat = detect_lonlat(plot_model)

    fig = pygmt.Figure()

    pygmt.config(
        FONT_TITLE="13p,Helvetica-Bold",
        FONT_LABEL="10p,Helvetica",
        FONT_ANNOT_PRIMARY="8p,Helvetica",
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xxx",
    )

    projection = "M16c"

    fig.basemap(
        region=region,
        projection=projection,
        frame=[
            "WSne+t00 Initiate model check",
            "xaf+lLongitude" if is_lonlat else "xaf+lX",
            "yaf+lLatitude" if is_lonlat else "yaf+lY",
        ],
    )

    plot_model_flyable_nofly(
        fig=fig,
        model=plot_model,
        model_marker_size=model_marker_size,
        model_alpha=model_alpha,
        no_fly_prefixes=plot_no_fly_prefixes,
        no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
        show_flz_overlay=plot_show_flz_overlay,
        always_flyable_prefixes=always_flyable_prefixes,
    )

    fig.plot(
        x=[start["x"]],
        y=[start["y"]],
        style="a0.45c",
        fill="yellow",
        pen="0.7p,black",
        label=f"Start: {start['label']}",
        transparency=50,
    )

    fig.plot(
        x=[end["x"]],
        y=[end["y"]],
        fill="grey@50",
        style="s0.42c",
        pen="1.4p,blue",
        label=f"End: {end['label']}",
        transparency=50,
    )

    fig.text(
        x=[start["x"], end["x"]],
        y=[start["y"], end["y"]],
        text=[str(start["label"]), str(end["label"])],
        font="9p,Helvetica-Bold,black",
        justify="LM",
        offset="0.15c/0.15c",
    )

    if is_lonlat:
        try:
            fig.basemap(map_scale="n0.50/0.06+c+w1k+f+l1 km")
        except Exception:
            pass

    fig.legend(
        position="JTL+jTL+o0.15c/0.15c",
        box="+gwhite@25+p0.5p,black",
    )

    fig.savefig(str(figure_file), dpi=dpi)
    return figure_file


# ============================================================
# Flyable / no-fly classification
# ============================================================

def classify_flyable_nofly(
    model: pd.DataFrame,
    no_fly_prefixes=("RA",),
    no_fly_slowness_threshold: float = 1e5,
    always_flyable_prefixes=("DB", "DK"),
):
    no_fly_prefixes = tuple(str(p).upper() for p in no_fly_prefixes)
    always_flyable_prefixes = tuple(str(p).upper() for p in always_flyable_prefixes)

    prefix = model["label_prefix"].astype(str).str.upper()

    prefix_mask = prefix.isin(no_fly_prefixes)
    slow_mask = (
        pd.to_numeric(model["slowness"], errors="coerce")
        .fillna(np.inf)
        >= float(no_fly_slowness_threshold)
    )

    always_flyable_mask = prefix.isin(always_flyable_prefixes)

    # No-fly by prefix/slowness, but DB/DK are always flyable exceptions.
    nofly_mask = (prefix_mask | slow_mask) & (~always_flyable_mask)
    flyable_mask = ~nofly_mask

    flyable = model[flyable_mask].copy()
    nofly = model[nofly_mask].copy()

    return flyable, nofly


def plot_model_flyable_nofly(
    fig: pygmt.Figure,
    model: pd.DataFrame,
    model_marker_size: float = 2.0,
    model_alpha: float = 0.45,
    no_fly_prefixes=("RA",),
    no_fly_slowness_threshold: float = 1e5,
    show_flz_overlay: bool = True,
    always_flyable_prefixes=("DB", "DK"),
):
    flyable, nofly = classify_flyable_nofly(
        model=model,
        no_fly_prefixes=no_fly_prefixes,
        no_fly_slowness_threshold=no_fly_slowness_threshold,
        always_flyable_prefixes=always_flyable_prefixes,
    )
    flyable_slow_text = get_representative_slowness_text(flyable)
    nofly_slow_text = get_representative_slowness_text(nofly)

    # Flyable nodes
    if not flyable.empty:
        fig.plot(
            x=flyable["x"],
            y=flyable["y"],
            style=f"c{max(model_marker_size / 35, 0.03):.3f}c",
            fill=f"seagreen@{alpha_to_transparency(model_alpha)}",
            pen=None,
            label=f"Flyable: slowness = {flyable_slow_text} (s/km)"
        )

    # No-fly nodes
    if not nofly.empty:
        fig.plot(
            x=nofly["x"],
            y=nofly["y"],
            style=f"s{max(model_marker_size / 28, 0.05):.3f}c",
            fill="red@20",
            pen="0.15p,red",
            label=f"No-fly: slowness = {nofly_slow_text} (s/km)"
        )

    # Optional FLZ overlay
    if show_flz_overlay:
        flz = model[model["label_prefix"].astype(str).str.upper() == "FLZ"].copy()
        if not flz.empty:
            fig.plot(
                x=flz["x"],
                y=flz["y"],
                style=f"c{max(model_marker_size / 18, 0.06):.3f}c",
                fill="orange@25",
                pen="0.2p,black",
                label=f"FLZ",
            )


# ============================================================
# Helpers
# ============================================================

def sample_model_for_plot(
    model: pd.DataFrame,
    max_model_points: int = 300000,
) -> pd.DataFrame:
    if len(model) <= max_model_points:
        return model.copy()

    special_prefixes = ["DB", "DK", "FLZ", "RA"]

    special = model[model["label_prefix"].isin(special_prefixes)].copy()
    normal = model[~model.index.isin(special.index)].copy()

    remain = max(max_model_points - len(special), 1000)

    if len(normal) > remain:
        normal = normal.sample(n=remain, random_state=42)

    out = pd.concat([normal, special], axis=0)
    out = out.sort_index()

    return out


def get_xy_region_with_padding(
    model: pd.DataFrame,
    path_df: pd.DataFrame,
    padding_ratio: float = 0.04,
) -> list[float]:
    xmin = min(model["x"].min(), path_df["x"].min())
    xmax = max(model["x"].max(), path_df["x"].max())
    ymin = min(model["y"].min(), path_df["y"].min())
    ymax = max(model["y"].max(), path_df["y"].max())

    dx = xmax - xmin
    dy = ymax - ymin

    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0

    padx = dx * padding_ratio
    pady = dy * padding_ratio

    return [
        float(xmin - padx),
        float(xmax + padx),
        float(ymin - pady),
        float(ymax + pady),
    ]


def get_profile_region(
    x,
    y,
    y_padding_ratio: float = 0.12,
) -> list[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    ymin = float(np.nanmin(y))
    ymax = float(np.nanmax(y))

    if xmin == xmax:
        xmin -= 1
        xmax += 1

    dy = ymax - ymin
    if dy == 0:
        dy = max(abs(ymin) * 0.1, 1.0)

    ymin -= dy * y_padding_ratio
    ymax += dy * y_padding_ratio

    return [xmin, xmax, ymin, ymax]


def detect_lonlat(model: pd.DataFrame) -> bool:
    xmin, xmax = model["x"].min(), model["x"].max()
    ymin, ymax = model["y"].min(), model["y"].max()

    return (
        -180 <= xmin <= 180
        and -180 <= xmax <= 180
        and -90 <= ymin <= 90
        and -90 <= ymax <= 90
    )


def alpha_to_transparency(alpha: float) -> int:
    alpha = max(0.0, min(1.0, float(alpha)))
    return int(round((1.0 - alpha) * 100))

def add_report_text_box(
    fig: pygmt.Figure,
    region: list[float],
    algorithm_name: str,
    path_df: pd.DataFrame,
    result: dict | None = None,
):
    """
    Add a text report box at the top-right of the main map.

    Uses fig.text with a semi-transparent white box.
    """

    if result is None:
        result = {}

    xmin, xmax, ymin, ymax = region

    # Top-right anchor position
    x_text = xmax - 0.02 * (xmax - xmin)
    y_text = ymax - 0.04 * (ymax - ymin)

    path_nodes = len(path_df)

    distance_km = result.get("output_path_distance_km", None)
    distance_m = result.get("output_path_distance_m", None)

    traveltime_s = result.get("output_estimated_traveltime_s", None)
    traveltime_min = result.get("output_estimated_traveltime_min", None)

    runtime_s = result.get("runtime_seconds", None)
    expanded_nodes = result.get("expanded_nodes", None)

    if distance_km is not None:
        distance_text = f"{float(distance_km):.3f} km"
    elif distance_m is not None:
        distance_text = f"{float(distance_m):.1f} m"
    else:
        distance_text = "N/A"

    if traveltime_min is not None:
        traveltime_text = f"{float(traveltime_min):.2f} min"
    elif traveltime_s is not None:
        traveltime_text = f"{float(traveltime_s):.1f} s"
    else:
        traveltime_text = "N/A"

    if runtime_s is not None:
        runtime_text = f"{float(runtime_s):.3f} s"
    else:
        runtime_text = "N/A"

    if expanded_nodes is not None:
        expanded_text = f"{int(expanded_nodes):,}"
    else:
        expanded_text = "N/A"

    text = (
        f"Algorithm: {algorithm_name}\n"
        f"Path nodes: {path_nodes:,}\n"
        f"Distance: {distance_text}\n"
        f"Travel time: {traveltime_text}\n"
        f"Expanded: {expanded_text}\n"
        f"Runtime: {runtime_text}"
    )

    fig.text(
        x=x_text,
        y=y_text,
        text=text,
        font="8.5p,Helvetica-Bold,black",
        justify="TR",
        fill="white@15",
        pen="0.5p,black",
        clearance="0.12c/0.12c",
    )

def add_path_traveltime_columns(
    model: pd.DataFrame,
    path_df: pd.DataFrame,
    path_indices: list[int],
) -> pd.DataFrame:
    """
    Add segment and cumulative traveltime columns for plotting.

    Traveltime estimate:
      distance_m * average slowness

    If slowness is relative cost, then cumulative_traveltime_s is relative cost.
    """

    out = path_df.copy()

    x = model.loc[path_indices, "x"].astype(float).values
    y = model.loc[path_indices, "y"].astype(float).values
    z = model.loc[path_indices, "z"].astype(float).values
    s = model.loc[path_indices, "slowness"].astype(float).values

    is_lonlat = (
        -180 <= np.nanmin(x) <= 180
        and -180 <= np.nanmax(x) <= 180
        and -90 <= np.nanmin(y) <= 90
        and -90 <= np.nanmax(y) <= 90
    )

    if is_lonlat:
        lon0 = float(np.nanmean(x))
        lat0 = float(np.nanmean(y))
        lat0_rad = np.deg2rad(lat0)

        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = 111_320.0 * np.cos(lat0_rad)

        xm = (x - lon0) * meters_per_deg_lon
        ym = (y - lat0) * meters_per_deg_lat
        zm = z
    else:
        xm = x
        ym = y
        zm = z

    segment_distance_m = np.zeros(len(out), dtype=float)
    segment_traveltime_s = np.zeros(len(out), dtype=float)

    positive_s = s[np.isfinite(s) & (s > 0)]
    if len(positive_s) > 0:
        min_positive_s = float(np.min(positive_s))
    else:
        min_positive_s = 1.0

    for i in range(1, len(out)):
        dx = xm[i] - xm[i - 1]
        dy = ym[i] - ym[i - 1]
        dz = zm[i] - zm[i - 1]

        dist_m = float(np.sqrt(dx * dx + dy * dy + dz * dz))

        s1 = float(s[i - 1])
        s2 = float(s[i])

        # DB/DK sometimes have slowness = 0. Use minimum positive slowness.
        if s1 <= 0 or not np.isfinite(s1):
            s1 = min_positive_s
        if s2 <= 0 or not np.isfinite(s2):
            s2 = min_positive_s

        avg_s = 0.5 * (s1 + s2)

        segment_distance_m[i] = dist_m
        segment_traveltime_s[i] = dist_m * avg_s

    out["segment_distance_m"] = segment_distance_m
    out["cumulative_distance_m"] = np.cumsum(segment_distance_m)
    out["cumulative_distance_km"] = out["cumulative_distance_m"] / 1000.0

    out["segment_traveltime_s"] = segment_traveltime_s
    out["cumulative_traveltime_s"] = np.cumsum(segment_traveltime_s)
    out["cumulative_traveltime_min"] = out["cumulative_traveltime_s"] / 60.0

    return out

def add_report_text_box_to_profile(
    fig: pygmt.Figure,
    region: list[float],
    algorithm_name: str,
    path_df: pd.DataFrame,
    result: dict | None = None,
):
    """
    Add report text box at the top-right of the current profile panel.
    """

    if result is None:
        return

    xmin, xmax, ymin, ymax = region

    path_nodes = len(path_df)

    distance_km = result.get("output_path_distance_km", None)
    distance_m = result.get("output_path_distance_m", None)

    traveltime_s = result.get("output_estimated_traveltime_s", None)
    traveltime_min = result.get("output_estimated_traveltime_min", None)

    runtime_s = result.get("runtime_seconds", None)
    expanded_nodes = result.get("expanded_nodes", None)

    if distance_km is not None:
        distance_text = f"{float(distance_km):.3f} km"
    elif distance_m is not None:
        distance_text = f"{float(distance_m):.1f} m"
    else:
        distance_text = f"{float(path_df['cumulative_distance_km'].iloc[-1]):.3f} km"

    if traveltime_min is not None:
        traveltime_text = f"{float(traveltime_min):.2f} min"
    elif traveltime_s is not None:
        traveltime_text = f"{float(traveltime_s):.1f} s"
    else:
        traveltime_text = f"{float(path_df['cumulative_traveltime_s'].iloc[-1]):.1f} s"

    if runtime_s is not None:
        runtime_text = f"{float(runtime_s):.3f} s"
    else:
        runtime_text = "N/A"

    if expanded_nodes is not None:
        expanded_text = f"{int(expanded_nodes):,}"
    else:
        expanded_text = "N/A"

    text = (
        f"Algorithm: {algorithm_name} | Path nodes: {path_nodes:,} | "
        f"Distance: {distance_text}"
    )

    text1 = (
        f"Travel time: {traveltime_text} | Expanded: {expanded_text} | "
        f"Runtime: {runtime_text}"
    )

    # Move all report text boxes to the left.
    # Increase 0.08 to 0.10 or 0.12 if still too far right.
    x_text = xmax - 0.15 * (xmax - xmin)

    # Two separated lines
    y_text0 = ymax - 0.10 * (ymax - ymin)
    y_text1 = ymax - 0.22 * (ymax - ymin)

    fig.text(
        x=x_text,
        y=y_text0,
        text=text,
        font="8p,Helvetica-Bold,black",
        justify="TR",
        fill="white@50",
        pen="0.4p,black",
        clearance="0.10c/0.10c",
    )

    fig.text(
        x=x_text,
        y=y_text1,
        text=text1,
        font="8p,Helvetica-Bold,black",
        justify="TR",
        fill="white@50",
        pen="0.4p,black",
        clearance="0.10c/0.10c",
    )

def get_representative_slowness_text(df: pd.DataFrame) -> str:
    """
    Return representative slowness value for legend.

    If all values are the same:
      0.02

    If multiple values exist:
      min-max
    """

    if df is None or df.empty:
        return "N/A"

    slow = pd.to_numeric(df["slowness"], errors="coerce")
    slow = slow[np.isfinite(slow)]

    if len(slow) == 0:
        return "N/A"

    smin = float(slow.min())
    smax = float(slow.max())

    if np.isclose(smin, smax):
        return format_slowness_value(smin)

    return f"{format_slowness_value(smax)}"


def format_slowness_value(value: float) -> str:
    """
    Nice formatting for slowness legend.
    """

    value = float(value)

    if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:.1e}"

    if abs(value) < 1:
        return f"{value:.3g}"

    if float(value).is_integer():
        return f"{value:.0f}"

    return f"{value:.3g}"
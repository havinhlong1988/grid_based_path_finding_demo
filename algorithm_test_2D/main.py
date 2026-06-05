#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main controller for Scenario 1 path finding.

Flow:
  1. Read parameters from parameters.py
  2. Load labelled model XYZ
  3. Select real start/end nodes, usually DB/DK
  4. Plot initiate model figure
  5. Build graph
  6. Snap real DB/DK to searchable grid nodes if needed
  7. Force search start/end and endpoint buffer as flyable if requested
  8. Run selected algorithm
  9. Export path footprint files
 10. Plot model + path report
 11. Optional cleanup

Expected outputs:
  output/dat/senario1/{algorithm}/path_senario1_{algorithm}.csv
  output/dat/senario1/{algorithm}/path_senario1_{algorithm}.xyz
  output/dat/senario1/{algorithm}/path_senario1_{algorithm}.geojson
  output/dat/senario1/{algorithm}/path_senario1_{algorithm}.kml
  output/dat/senario1/{algorithm}/path_senario1_{algorithm}_summary.json

  output/figures/senario1/00_initiate.png
  output/figures/senario1/path_report_{algorithm}.png
"""

from pathlib import Path
import importlib
import json
import sys

try:
    import parameters as parameter
except ModuleNotFoundError:
    import parameter

from src.model_io import *

from src.output_io import export_path_outputs
from src.plotting import plot_path_report, plot_initiate_model
from src.cleanup import cleanup_intermediate_files, print_cleanup_summary


def get_param(name, default=None):
    return getattr(parameter, name, default)


def main():
    # ============================================================
    # Basic paths and algorithm
    # ============================================================
    model_file = Path(
        get_param(
            "MODEL_FILE",
            Path("model") / "senario1" / "model_senario1_with_label.xyz",
        )
    )

    algorithm_name = str(get_param("ALGORITHM", "astar")).lower()

    dat_root_dir = Path(
        get_param(
            "DAT_ROOT_DIR",
            Path("output") / "dat" / "senario1",
        )
    )

    figure_root_dir = Path(
        get_param(
            "FIGURE_ROOT_DIR",
            Path("output") / "figures" / "senario1",
        )
    )

    # Data/report files:
    # output/dat/senario1/{algorithm}/
    output_dir = dat_root_dir / algorithm_name

    # Figure files:
    # output/figures/senario1/
    # Algorithm-specific figure folder:
    # output/figures/senario1/{algorithm}/
    algorithm_figure_dir = figure_root_dir / algorithm_name



    # ============================================================
    # Start/end settings
    # ============================================================
    start_label = get_param("START_LABEL", None)
    end_label = get_param("END_LABEL", None)
    start_coord = get_param("START_COORD", None)
    end_coord = get_param("END_COORD", None)

    # Figure files:
    # output/figures/senario1/{algorithm}/
    safe_start_label = str(start_label).replace("/", "_").replace("\\", "_").replace(" ", "_")
    safe_end_label = str(end_label).replace("/", "_").replace("\\", "_").replace(" ", "_")

    report_figure_file = (
        algorithm_figure_dir
        / f"path_report_{algorithm_name}_from_{safe_start_label}_to_{safe_end_label}.png"
    )
    
    snap_start_end_to_grid = bool(get_param("SNAP_START_END_TO_GRID", True))
    snap_target_prefixes = tuple(get_param("SNAP_TARGET_PREFIXES", ("N", "FLZ")))
    include_real_start_end = bool(get_param("INCLUDE_REAL_START_END_IN_OUTPUT", True))

    always_flyable_prefixes = tuple(get_param("ALWAYS_FLYABLE_PREFIXES", ("DB", "DK")))
    force_search_start_end_flyable = bool(
        get_param("FORCE_SEARCH_START_END_FLYABLE", True)
    )

    endpoint_flyable_buffer_radius_m = float(
        get_param("ENDPOINT_FLYABLE_BUFFER_RADIUS_M", 0.0)
    )
    endpoint_flyable_buffer_mode = str(
        get_param("ENDPOINT_FLYABLE_BUFFER_MODE", "both")
    ).lower()

    # ============================================================
    # Graph rules
    # ============================================================
    block_prefixes = tuple(get_param("BLOCK_LABEL_PREFIXES", ("RA",)))
    high_cost_prefixes = tuple(get_param("HIGH_COST_LABEL_PREFIXES", ("FLZ",)))
    flz_cost_factor = float(get_param("FLZ_COST_FACTOR", 5.0))

    connectivity_2d = int(get_param("CONNECTIVITY_2D", 8))
    connectivity_3d = int(get_param("CONNECTIVITY_3D", 26))

    graph_neighbor_mode = str(get_param("GRAPH_NEIGHBOR_MODE", "kdtree")).lower()
    kdtree_radius_factor = float(get_param("KDTREE_RADIUS_FACTOR", 1.60))
    kdtree_max_neighbors_2d = int(get_param("KDTREE_MAX_NEIGHBORS_2D", 8))
    kdtree_max_neighbors_3d = int(get_param("KDTREE_MAX_NEIGHBORS_3D", 26))

    # ============================================================
    # Output naming
    # ============================================================
    path_name = str(get_param("PATH_NAME", "path_senario1"))

    # ============================================================
    # Plot settings
    # ============================================================
    plot_max_model_points = int(get_param("PLOT_MAX_MODEL_POINTS", 300000))
    plot_dpi = int(get_param("PLOT_DPI", 300))
    plot_model_alpha = float(get_param("PLOT_MODEL_ALPHA", 0.45))
    plot_model_marker_size = float(get_param("PLOT_MODEL_MARKER_SIZE", 2.0))
    plot_path_line_width = float(get_param("PLOT_PATH_LINE_WIDTH", 2.0))

    plot_initiate_figure = bool(get_param("PLOT_INITIATE_FIGURE", True))
    initiate_figure_name = str(get_param("INITIATE_FIGURE_NAME", "00_initiate.png"))
    initiate_figure_file = algorithm_figure_dir / initiate_figure_name

    plot_model_as_flyable_nofly = bool(
        get_param("PLOT_MODEL_AS_FLYABLE_NOFLY", True)
    )
    plot_no_fly_prefixes = tuple(get_param("PLOT_NO_FLY_PREFIXES", ("RA",)))
    plot_no_fly_slowness_threshold = float(
        get_param("PLOT_NO_FLY_SLOWNESS_THRESHOLD", 1e5)
    )
    cap_slowness_after_load = bool(get_param("CAP_SLOWNESS_AFTER_LOAD", True))
    slowness_cap_value = float(
        get_param("SLOWNESS_CAP_VALUE", plot_no_fly_slowness_threshold)
    )
    plot_show_flz_overlay = bool(get_param("PLOT_SHOW_FLZ_OVERLAY", True))
    plot_always_flyable_prefixes = tuple(
        get_param("PLOT_ALWAYS_FLYABLE_PREFIXES", ("DB", "DK"))
    )
    plot_report_text_box = bool(get_param("PLOT_REPORT_TEXT_BOX", True))
    # ============================================================
    # Cleanup settings
    # ============================================================
    run_cleanup = bool(get_param("RUN_CLEANUP", False))
    cleanup_dry_run = bool(get_param("CLEANUP_DRY_RUN", True))
    cleanup_empty_dirs = bool(get_param("CLEANUP_EMPTY_DIRS", True))
    cleanup_target_dirs = get_param(
        "CLEANUP_TARGET_DIRS",
        [dat_root_dir, figure_root_dir],
    )
    cleanup_patterns = get_param(
        "CLEANUP_PATTERNS",
        [
            "*.tmp",
            "*.temp",
            "*.bak",
            "*.backup",
            "*.log",
            "*.cache",
            "*.gmt",
            "*.cpt",
            "*.grd",
            "*.nc",
            "*.vrt",
            "*.aux.xml",
            "*_tmp.*",
            "*_temp.*",
            "tmp_*",
            "temp_*",
            ".gmt*",
        ],
    )

    # ============================================================
    # Make folders
    # ============================================================
    dat_root_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_root_dir.mkdir(parents=True, exist_ok=True)
    algorithm_figure_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Header
    # ============================================================
    print("=" * 70)
    print("PATH-FINDING MAIN CONTROLLER")
    print("=" * 70)
    print(f"Model file       : {model_file}")
    print(f"Algorithm        : {algorithm_name}")
    print(f"Data output dir  : {output_dir}")
    print(f"Figure dir       : {algorithm_figure_dir}")
    print(f"Report figure    : {report_figure_file}")
    print(f"Initiate figure  : {initiate_figure_file}")
    print(f"Run cleanup      : {run_cleanup}")
    print("=" * 70)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # ============================================================
    # 1. Load model
    # ============================================================
    print("[1/6] Loading labelled model...")
    model = load_labelled_model(model_file)

    print(f"      Nodes loaded: {len(model):,}")
    print(f"      Columns     : {list(model.columns)}")

    if cap_slowness_after_load:
        model, cap_summary = cap_slowness_values(
            model=model,
            cap_value=slowness_cap_value,
            inplace=False,
        )

        print("      Slowness cap:")
        print(f"        cap value       : {cap_summary['cap_value']:.6g}")
        print(f"        capped nodes    : {cap_summary['n_capped']:,}")
        print(f"        old max slowness: {cap_summary['old_max_slowness']:.6g}")
        print(f"        new max slowness: {cap_summary['new_max_slowness']:.6g}")

    # ============================================================
    # 2. Select real start/end
    # ============================================================
    print("[2/6] Selecting start/end nodes...")
    start_idx, end_idx = find_start_end_indices(
        model,
        start_label=start_label,
        end_label=end_label,
        start_coord=start_coord,
        end_coord=end_coord,
    )

    print(f"      Start index: {start_idx}")
    print(f"      End index  : {end_idx}")
    print(f"      Start node : {model.loc[start_idx].to_dict()}")
    print(f"      End node   : {model.loc[end_idx].to_dict()}")

    # ============================================================
    # INIT. Plot initiate figure
    # ============================================================
    if plot_initiate_figure:
        print("[INIT] Plotting initiate model figure...")
        plot_initiate_model(
            model=model,
            start_idx=start_idx,
            end_idx=end_idx,
            figure_file=initiate_figure_file,
            max_model_points=plot_max_model_points,
            dpi=plot_dpi,
            model_alpha=plot_model_alpha,
            model_marker_size=plot_model_marker_size,
            plot_model_as_flyable_nofly=plot_model_as_flyable_nofly,
            plot_no_fly_prefixes=plot_no_fly_prefixes,
            plot_no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
            plot_show_flz_overlay=plot_show_flz_overlay,
            always_flyable_prefixes=plot_always_flyable_prefixes,
        )
        print(f"      Initiate figure: {initiate_figure_file}")

    # ============================================================
    # 3. Build graph
    # ============================================================
    print("[3/6] Building graph...")
    graph = build_grid_graph(
        model,
        block_label_prefixes=block_prefixes,
        high_cost_label_prefixes=high_cost_prefixes,
        high_cost_factor=flz_cost_factor,
        connectivity_2d=connectivity_2d,
        connectivity_3d=connectivity_3d,
        always_flyable_prefixes=always_flyable_prefixes,
        graph_neighbor_mode=graph_neighbor_mode,
        kdtree_radius_factor=kdtree_radius_factor,
        kdtree_max_neighbors_2d=kdtree_max_neighbors_2d,
        kdtree_max_neighbors_3d=kdtree_max_neighbors_3d,
    )

    print(f"      Traversable nodes: {len(graph['valid_indices']):,}")
    print(f"      Grid dimension   : {graph['dimension']}D")
    print(f"      Connectivity     : {graph['connectivity']}")

    if "graph_neighbor_mode" in graph:
        print(f"      Neighbor mode    : {graph['graph_neighbor_mode']}")
    if "grid_spacing_m" in graph:
        print(f"      Grid spacing     : {graph['grid_spacing_m']:.2f} m")
    if "neighbor_radius_m" in graph:
        print(f"      Neighbor radius  : {graph['neighbor_radius_m']:.2f} m")
    if "max_neighbors" in graph:
        print(f"      Max neighbors    : {graph['max_neighbors']}")

    # ============================================================
    # Snap DB/DK or special points to grid/search nodes
    # ============================================================
    search_start_idx, search_end_idx = snap_start_end_to_grid_if_needed(
        model=model,
        graph=graph,
        start_idx=start_idx,
        end_idx=end_idx,
        snap=snap_start_end_to_grid,
        target_prefixes=snap_target_prefixes,
    )

    # ============================================================
    # Force search endpoints to be flyable
    # ============================================================
    if force_search_start_end_flyable:
        graph["valid_indices"].add(int(search_start_idx))
        graph["valid_indices"].add(int(search_end_idx))

        print("      Force search start/end as flyable:")
        print(
            f"        search start: {search_start_idx} | "
            f"{model.loc[search_start_idx, 'label']}"
        )
        print(
            f"        search end  : {search_end_idx} | "
            f"{model.loc[search_end_idx, 'label']}"
        )

    # ============================================================
    # Endpoint flyable buffer
    # ============================================================
    if endpoint_flyable_buffer_radius_m > 0:
        buffer_endpoint_indices = []

        if endpoint_flyable_buffer_mode in ("real", "both"):
            buffer_endpoint_indices.extend([start_idx, end_idx])

        if endpoint_flyable_buffer_mode in ("search", "both"):
            buffer_endpoint_indices.extend([search_start_idx, search_end_idx])

        # Remove duplicates but keep order
        buffer_endpoint_indices = list(
            dict.fromkeys([int(i) for i in buffer_endpoint_indices])
        )

        before_n_valid = len(graph["valid_indices"])

        graph = add_endpoint_flyable_buffer(
            model=model,
            graph=graph,
            endpoint_indices=buffer_endpoint_indices,
            radius_m=endpoint_flyable_buffer_radius_m,
        )

        after_n_valid = len(graph["valid_indices"])
        added_n = after_n_valid - before_n_valid

        print("      Endpoint flyable buffer:")
        print(f"        radius           : {endpoint_flyable_buffer_radius_m:.2f} m")
        print(f"        mode             : {endpoint_flyable_buffer_mode}")
        print(f"        endpoint indices : {buffer_endpoint_indices}")
        print(f"        added flyable    : {added_n:,} nodes")

    # ============================================================
    # Print snapping information
    # ============================================================
    if search_start_idx != start_idx:
        print("      Start snapped:")
        print(
            f"        real start index  : {start_idx} | "
            f"{model.loc[start_idx, 'label']}"
        )
        print(
            f"        search start index: {search_start_idx} | "
            f"{model.loc[search_start_idx, 'label']}"
        )

    if search_end_idx != end_idx:
        print("      End snapped:")
        print(
            f"        real end index  : {end_idx} | "
            f"{model.loc[end_idx, 'label']}"
        )
        print(
            f"        search end index: {search_end_idx} | "
            f"{model.loc[search_end_idx, 'label']}"
        )

    # ============================================================
    # Neighbor diagnostic
    # ============================================================
    print("      Neighbor diagnostic:")
    start_n_neighbors = count_valid_neighbors(model, graph, search_start_idx)
    end_n_neighbors = count_valid_neighbors(model, graph, search_end_idx)

    print(f"        search start neighbors: {start_n_neighbors}")
    print(f"        search end neighbors  : {end_n_neighbors}")

    if start_n_neighbors == 0 or end_n_neighbors == 0:
        print("[WARNING] Search start or end has zero graph neighbors.")
        print("          Try increasing KDTREE_RADIUS_FACTOR in parameters.py, for example:")
        print("          KDTREE_RADIUS_FACTOR = 2.0")

    print(f"      Traversable nodes: {len(graph['valid_indices']):,}")
    print(f"      Grid dimension   : {graph['dimension']}D")
    print(f"      Connectivity     : {graph['connectivity']}")

    # ============================================================
    # 4. Run algorithm
    # ============================================================
    print("[4/6] Running algorithm...")
    try:
        alg_module = importlib.import_module(f"src.{algorithm_name}")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Algorithm src/{algorithm_name}.py not found. "
            f"Available examples: astar, flood_fill"
        ) from exc

    if not hasattr(alg_module, "run"):
        raise AttributeError(
            f"src/{algorithm_name}.py must contain a run(...) function."
        )

    result = alg_module.run(
        model=model,
        graph=graph,
        start_idx=search_start_idx,
        end_idx=search_end_idx,
    )

    algorithm_path_indices = result.get("path_indices", [])

    if not algorithm_path_indices:
        print("[FAILED] No path found.")

        result["real_start_idx"] = int(start_idx)
        result["real_end_idx"] = int(end_idx)
        result["search_start_idx"] = int(search_start_idx)
        result["search_end_idx"] = int(search_end_idx)
        result["include_real_start_end_in_output"] = bool(include_real_start_end)
        result["endpoint_flyable_buffer_radius_m"] = float(
            endpoint_flyable_buffer_radius_m
        )
        result["endpoint_flyable_buffer_mode"] = str(endpoint_flyable_buffer_mode)

        # Add slowness-cap metadata
        result["cap_slowness_after_load"] = bool(cap_slowness_after_load)
        result["slowness_cap_value"] = float(slowness_cap_value)

        result["start_neighbors"] = int(start_n_neighbors)
        result["end_neighbors"] = int(end_n_neighbors)

        fail_file = output_dir / f"{path_name}_{algorithm_name}_FAILED.json"
        fail_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

        print(f"Failure summary saved to: {fail_file}")
        sys.exit(1)

    # Add real DB/DK to output footprint if needed
    path_indices = add_real_start_end_to_path(
        path_indices=algorithm_path_indices,
        real_start_idx=start_idx,
        real_end_idx=end_idx,
        include=include_real_start_end,
    )
    # Compute metrics for algorithm path only.
    # This avoids counting artificial DB/DK connector segments if DB/DK are off-grid.
    algorithm_path_metrics = compute_path_metrics(
        model=model,
        graph=graph,
        path_indices=algorithm_path_indices,
    )

    # Compute metrics for exported path including DB/DK if included.
    # This gives the full footprint distance from real DB to real DK.
    output_path_metrics = compute_path_metrics(
        model=model,
        graph=graph,
        path_indices=path_indices,
    )

    # Add metadata
    result["real_start_idx"] = int(start_idx)
    result["real_end_idx"] = int(end_idx)
    result["search_start_idx"] = int(search_start_idx)
    result["search_end_idx"] = int(search_end_idx)
    result["include_real_start_end_in_output"] = bool(include_real_start_end)
    result["endpoint_flyable_buffer_radius_m"] = float(
        endpoint_flyable_buffer_radius_m
    )
    result["endpoint_flyable_buffer_mode"] = str(endpoint_flyable_buffer_mode)

    # Add slowness-cap metadata
    result["cap_slowness_after_load"] = bool(cap_slowness_after_load)
    result["slowness_cap_value"] = float(slowness_cap_value)

    result["start_neighbors"] = int(start_n_neighbors)
    result["end_neighbors"] = int(end_n_neighbors)

    result["graph_neighbor_mode"] = str(graph.get("graph_neighbor_mode", "unknown"))
    result["grid_spacing_m"] = float(graph.get("grid_spacing_m", 0.0))
    result["neighbor_radius_m"] = float(graph.get("neighbor_radius_m", 0.0))
    result["max_neighbors"] = int(graph.get("max_neighbors", 0))

    result["algorithm_path_distance_m"] = algorithm_path_metrics["distance_traveled_m"]
    result["algorithm_path_distance_km"] = algorithm_path_metrics["distance_traveled_km"]
    result["algorithm_estimated_traveltime_s"] = algorithm_path_metrics["estimated_traveltime_s"]
    result["algorithm_estimated_traveltime_min"] = algorithm_path_metrics["estimated_traveltime_min"]

    result["output_path_distance_m"] = output_path_metrics["distance_traveled_m"]
    result["output_path_distance_km"] = output_path_metrics["distance_traveled_km"]
    result["output_estimated_traveltime_s"] = output_path_metrics["estimated_traveltime_s"]
    result["output_estimated_traveltime_min"] = output_path_metrics["estimated_traveltime_min"]

    print(f"      Algorithm path nodes       : {len(algorithm_path_indices):,}")
    print(f"      Output path nodes          : {len(path_indices):,}")
    print(f"      Total cost                 : {result.get('total_cost', None)}")

    print("      Path metrics:")
    print(f"        algorithm distance       : {algorithm_path_metrics['distance_traveled_m']:.2f} m")
    print(f"        algorithm distance       : {algorithm_path_metrics['distance_traveled_km']:.4f} km")
    print(f"        algorithm traveltime     : {algorithm_path_metrics['estimated_traveltime_s']:.2f} s")
    print(f"        algorithm traveltime     : {algorithm_path_metrics['estimated_traveltime_min']:.2f} min")

    print(f"        output distance          : {output_path_metrics['distance_traveled_m']:.2f} m")
    print(f"        output distance          : {output_path_metrics['distance_traveled_km']:.4f} km")
    print(f"        output traveltime        : {output_path_metrics['estimated_traveltime_s']:.2f} s")
    print(f"        output traveltime        : {output_path_metrics['estimated_traveltime_min']:.2f} min")

    # ============================================================
    # 5. Export path footprint files
    # ============================================================
    print("[5/6] Exporting path footprint files...")
    exported = export_path_outputs(
        model=model,
        path_indices=path_indices,
        output_dir=output_dir,
        path_name=f"{path_name}_{algorithm_name}",
        algorithm_name=algorithm_name,
        result=result,
    )

    # ============================================================
    # 6. Plot path report
    # ============================================================
    print("[6/6] Plotting path report...")
    plot_path_report(
        model=model,
        path_indices=path_indices,
        figure_file=report_figure_file,
        algorithm_name=algorithm_name,
        max_model_points=plot_max_model_points,
        dpi=plot_dpi,
        model_alpha=plot_model_alpha,
        model_marker_size=plot_model_marker_size,
        path_line_width=plot_path_line_width,
        plot_model_as_flyable_nofly=plot_model_as_flyable_nofly,
        plot_no_fly_prefixes=plot_no_fly_prefixes,
        plot_no_fly_slowness_threshold=plot_no_fly_slowness_threshold,
        plot_show_flz_overlay=plot_show_flz_overlay,
        always_flyable_prefixes=plot_always_flyable_prefixes,
        result=result if plot_report_text_box else None,
    )

    # ============================================================
    # Optional cleanup
    # ============================================================
    if run_cleanup:
        print("[CLEANUP] Removing intermediate files...")
        cleanup_summary = cleanup_intermediate_files(
            target_dirs=cleanup_target_dirs,
            patterns=cleanup_patterns,
            dry_run=cleanup_dry_run,
            remove_empty_dirs=cleanup_empty_dirs,
        )
        print_cleanup_summary(cleanup_summary)

    # ============================================================
    # Final print
    # ============================================================
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print("Exported path files:")
    for key, value in exported.items():
        print(f"  {key:12s}: {value}")

    print(f"Initiate figure : {initiate_figure_file}")
    print(f"Report figure   : {report_figure_file}")


if __name__ == "__main__":
    main()
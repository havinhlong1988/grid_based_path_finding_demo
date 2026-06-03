#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot downloaded Hoa Lac XYZ/GPKG data using PyGMT.

Input folder:
    output_hoalac_hitech_park/

Output folder:
    figures/01_download_xyz/

Outputs:
    00_overview_map.png
        Road map + building polygons + Hoa Lac boundary

    01_<xyz_filename>.png
        Individual XYZ plots

XYZ format:
    lon lat value
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt


# ============================================================
# 0.0.0 USER INPUT PARAMETERS
# ============================================================

DATA_DIR = Path("output_hoalac_hitech_park")
XYZ_DIR = DATA_DIR
FIG_DIR = Path("figures/01_download_xyz")


# Overview map output name
OVERVIEW_FIG_NAME = "00_overview_map.png"

# Input vector files for overview map
BUILDINGS_GPKG = DATA_DIR / "buildings_hoalac_clipped.gpkg"
ROADS_GPKG = DATA_DIR / "roads_hoalac_clipped.gpkg"

# Hoa Lac polygon
HOALAC_POLYGON = [
    (105.5035, 21.0145),
    (105.5125, 20.9935),
    (105.5310, 20.9815),
    (105.5565, 20.9845),
    (105.5735, 20.9985),
    (105.5705, 21.0190),
    (105.5480, 21.0285),
    (105.5205, 21.0270),
    (105.5035, 21.0145),
]

# Figure setting
PROJECTION = "M15c"
POINT_SIZE = "0.05c"

# Region padding in degree
REGION_PADDING = 0.003

# Plot all xyz files in input directory
XYZ_PATTERN = "*.xyz"

# ============================================================
# 0.0.1 MAP STYLE PARAMETERS
# ============================================================

# ----------------------------
# Hoa Lac polygon boundary
# ----------------------------
POLYGON_LINE_WIDTH = "1.4p"
POLYGON_LINE_COLOR = "purple"
POLYGON_LINE_STYLE = "-"       # "-" solid, "." dotted, "--" dashed
POLYGON_PEN = f"{POLYGON_LINE_WIDTH},{POLYGON_LINE_COLOR},{POLYGON_LINE_STYLE}"

# ----------------------------
# Road style
# ----------------------------
ROAD_LINE_WIDTH = "0.5p"
ROAD_LINE_COLOR = "gray40"
ROAD_LINE_STYLE = None          # "-" solid, "." dotted, "--" dashed
ROAD_PEN = f"{ROAD_LINE_WIDTH},{ROAD_LINE_COLOR},{ROAD_LINE_STYLE}"

# Road style for individual road XYZ figure
ROAD_XYZ_LINE_WIDTH = "0.5p"
ROAD_XYZ_LINE_COLOR = "gray40"
ROAD_XYZ_LINE_STYLE = None
ROAD_XYZ_PEN = f"{ROAD_XYZ_LINE_WIDTH},{ROAD_XYZ_LINE_COLOR},{ROAD_XYZ_LINE_STYLE}"

# ----------------------------
# Building polygon style
# ----------------------------
BUILDING_FILL_COLOR = "lightred"
BUILDING_FILL_TRANSPARENCY = 60     # 0 opaque, 100 fully transparent
BUILDING_FILL = f"{BUILDING_FILL_COLOR}@{BUILDING_FILL_TRANSPARENCY}"

BUILDING_PEN_WIDTH = "0.25p"
BUILDING_PEN_COLOR = "black"
BUILDING_PEN_STYLE = None
BUILDING_PEN_TRANSPARENCY = 20
BUILDING_PEN = f"{BUILDING_PEN_WIDTH},{BUILDING_PEN_COLOR}@{BUILDING_PEN_TRANSPARENCY},{BUILDING_PEN_STYLE}"

# ----------------------------
# Legend style
# ----------------------------
LEGEND_BOX_FILL = "white@10"
LEGEND_BOX_PEN = "0.5p,black"
LEGEND_POSITION = "JBL+jBL+o0.2c/0.2c"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_polygon_region(polygon, padding=0.003):
    """
    Get plot region from polygon.
    """
    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]

    west = min(lons) - padding
    east = max(lons) + padding
    south = min(lats) - padding
    north = max(lats) + padding

    return [west, east, south, north]


def read_xyz(xyz_file, keep_nan=False):
    """
    Read XYZ file.

    Expected format:
        lon lat value

    keep_nan=True is useful for road XYZ files because NaN rows
    are used as line-segment separators.
    """
    xyz_file = Path(xyz_file)

    if xyz_file.stat().st_size == 0:
        return None

    df = pd.read_csv(
        xyz_file,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "value"],
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    if keep_nan:
        if df.empty:
            return None
        return df

    df = df.dropna(subset=["lon", "lat", "value"])

    if df.empty:
        return None

    return df


def guess_plot_info(filename):
    """
    Guess colorbar label and colormap from filename.
    """
    name = filename.lower()

    if "dem" in name:
        cmap = "geo"
        label = "Elevation (m)"
    elif "slope" in name:
        cmap = "turbo"
        label = "Slope (degree)"
    elif "ruggedness" in name or "tri" in name:
        cmap = "oleron"
        label = "TRI (m)"
    elif "building" in name:
        cmap = "batlow"
        label = "Building height (m)"
    elif "road" in name:
        cmap = "categorical"
        label = "Road class code"
    else:
        cmap = "viridis"
        label = "Value"

    return cmap, label


def robust_zrange(values, lower=2, upper=98):
    """
    Robust color range using percentile.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0, 1.0

    zmin = np.percentile(values, lower)
    zmax = np.percentile(values, upper)

    if np.isclose(zmin, zmax):
        zmin = np.nanmin(values)
        zmax = np.nanmax(values)

    if np.isclose(zmin, zmax):
        zmin -= 1.0
        zmax += 1.0

    return float(zmin), float(zmax)


def plot_hoalac_boundary(fig):
    """
    Plot Hoa Lac polygon boundary.
    """
    poly_lons = [p[0] for p in HOALAC_POLYGON]
    poly_lats = [p[1] for p in HOALAC_POLYGON]

    fig.plot(
        x=poly_lons,
        y=poly_lats,
        pen=POLYGON_PEN,
    )


# ============================================================
# PLOT INDIVIDUAL XYZ FILES
# ============================================================

def plot_xyz_points(xyz_file, df, region, out_png):
    """
    Plot XYZ points using PyGMT.
    """
    cmap, label = guess_plot_info(xyz_file.name)

    zmin, zmax = robust_zrange(df["value"].to_numpy())

    # For building grid, many values are 0.
    # Use full max but keep 0 visible.
    if "building" in xyz_file.name.lower():
        zmin = 0.0
        zmax = max(float(df["value"].max()), 1.0)

    cpt_step = (zmax - zmin) / 100.0

    if cpt_step <= 0 or not np.isfinite(cpt_step):
        cpt_step = 1.0

    fig = pygmt.Figure()

    pygmt.makecpt(
        cmap=cmap,
        series=[zmin, zmax, cpt_step],
        continuous=True,
    )

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            "WSen",
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )

    fig.plot(
        x=df["lon"],
        y=df["lat"],
        style=f"c{POINT_SIZE}",
        fill=df["value"],
        cmap=True,
        pen=None,
    )

    plot_hoalac_boundary(fig)

    fig.colorbar(
        frame=f'af+l"{label}"',
        position="JBC+w10c/0.35c+h+o0c/0.8c",
    )

    fig.text(
        x=region[0],
        y=region[3],
        text=xyz_file.stem,
        font="12p,Helvetica-Bold,black",
        justify="TL",
        offset="0.1c/-0.1c",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"[OK] Saved figure: {out_png}")


def plot_roads_xyz(xyz_file, df, region, out_png):
    """
    Plot road map.

    Instead of connecting road vertices from XYZ, this function reads
    roads_hoalac_clipped.gpkg and plots each LineString separately.
    This avoids wrong connections between different road segments.
    """

    print("[INFO] Plotting roads from GPKG, not from XYZ vertices")

    roads = None

    if ROADS_GPKG.exists():
        roads = gpd.read_file(ROADS_GPKG)
        print(f"[INFO] Roads loaded from GPKG: {len(roads)}")
    else:
        print(f"[WARN] Road GPKG not found: {ROADS_GPKG}")
        print("[WARN] Fallback to XYZ road vertices may connect segments incorrectly.")

    fig = pygmt.Figure()

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            "WSen",
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )

    if roads is not None and not roads.empty:
        plot_lines_from_gdf(
            fig,
            roads,
            pen=ROAD_XYZ_PEN,
        )
    else:
        # Fallback only if GPKG is missing
        segment = []

        for _, row in df.iterrows():
            lon = row["lon"]
            lat = row["lat"]

            if not np.isfinite(lon) or not np.isfinite(lat):
                if len(segment) >= 2:
                    seg = np.asarray(segment)
                    fig.plot(
                        x=seg[:, 0],
                        y=seg[:, 1],
                        pen=ROAD_XYZ_PEN,
                    )
                segment = []
            else:
                segment.append([lon, lat])

        if len(segment) >= 2:
            seg = np.asarray(segment)
            fig.plot(
                x=seg[:, 0],
                y=seg[:, 1],
                pen=ROAD_XYZ_PEN,
            )

    plot_hoalac_boundary(fig)

    fig.text(
        x=region[0],
        y=region[3],
        text=xyz_file.stem,
        font="12p,Helvetica-Bold,black",
        justify="TL",
        offset="0.1c/-0.1c",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"[OK] Saved road figure: {out_png}")


# ============================================================
# OVERVIEW MAP: ROADS + BUILDING POLYGONS
# ============================================================

def plot_lines_from_gdf(fig, gdf, pen=None):
    """
    Plot LineString / MultiLineString geometries from GeoDataFrame.
    """
    if pen is None:
        pen = ROAD_PEN

    if gdf is None or gdf.empty:
        return

    gdf = gdf.to_crs("EPSG:4326").copy()

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            coords = np.asarray(line.coords)
            if coords.shape[0] >= 2:
                fig.plot(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    pen=pen,
                )


def plot_polygons_from_gdf(
    fig,
    gdf,
    fill=None,
    pen=None,
):
    """
    Plot Polygon / MultiPolygon geometries from GeoDataFrame.
    """
    if fill is None:
        fill = BUILDING_FILL

    if pen is None:
        pen = BUILDING_PEN

    if gdf is None or gdf.empty:
        return

    gdf = gdf.to_crs("EPSG:4326").copy()

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            continue

        for poly in polygons:
            x, y = poly.exterior.xy
            fig.plot(
                x=list(x),
                y=list(y),
                fill=fill,
                pen=pen,
            )


def plot_overview_map(region, out_png):
    """
    Plot one overview map:
        building polygons + road lines + Hoa Lac boundary
    """
    print("\n[INFO] Creating overview map")

    buildings = None
    roads = None

    if BUILDINGS_GPKG.exists():
        buildings = gpd.read_file(BUILDINGS_GPKG)
        print(f"[INFO] Buildings loaded: {len(buildings)}")
    else:
        print(f"[WARN] Building GPKG not found: {BUILDINGS_GPKG}")

    if ROADS_GPKG.exists():
        roads = gpd.read_file(ROADS_GPKG)
        print(f"[INFO] Roads loaded: {len(roads)}")
    else:
        print(f"[WARN] Roads GPKG not found: {ROADS_GPKG}")

    fig = pygmt.Figure()

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            'WSen+t"Hoa Lac overview map: Buildings + Roads"',
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )

    # Building polygons first
    plot_polygons_from_gdf(
        fig,
        buildings,
        fill=BUILDING_FILL,
        pen=BUILDING_PEN,
    )

    plot_lines_from_gdf(
        fig,
        roads,
        pen=ROAD_PEN,
    )
    # Hoa Lac boundary on top
    plot_hoalac_boundary(fig)

    # Fake plot to legend 
    fig.plot(
        x=[-1000],
        y=[-1000],
        style="s0.25c",
        fill=BUILDING_FILL,
        pen=BUILDING_PEN,
        label="Building polygon",
    )

    fig.plot(
        x=[-1000],
        y=[-1000],
        pen=ROAD_PEN,
        label="Road",
    )

    fig.plot(
        x=[-1000],
        y=[-1000],
        pen=POLYGON_PEN,
        label="Hoa Lac boundary",
    )

    fig.legend(
        position=LEGEND_POSITION,
        box=f"+g{LEGEND_BOX_FILL}+p{LEGEND_BOX_PEN}",
    )

    # fig.text(
    #     x=region[0],
    #     y=region[3],
    #     text="Hoa Lac overview: roads and buildings",
    #     font="12p,Helvetica-Bold,black",
    #     justify="TL",
    #     offset="0.1c/-0.1c",
    # )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"[OK] Saved overview map: {out_png}")


# ============================================================
# MAIN
# ============================================================

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    region = get_polygon_region(
        HOALAC_POLYGON,
        padding=REGION_PADDING,
    )

    print("\n========== PLOT HOA LAC DATA ==========")
    print(f"[INFO] Input data dir: {DATA_DIR}")
    print(f"[INFO] Output fig dir: {FIG_DIR}")

    # --------------------------------------------------------
    # 0. Overview map
    # --------------------------------------------------------
    overview_png = FIG_DIR / OVERVIEW_FIG_NAME

    plot_overview_map(
        region=region,
        out_png=overview_png,
    )

    # --------------------------------------------------------
    # 1. Individual XYZ plots
    # --------------------------------------------------------
    xyz_files = sorted(
        XYZ_DIR.glob(XYZ_PATTERN),
        key=lambda p: p.stat().st_mtime,
    )

    if len(xyz_files) == 0:
        raise FileNotFoundError(f"No XYZ files found in: {XYZ_DIR}")

    print("\n========== PLOT XYZ FILES ==========")
    print(f"[INFO] Number of XYZ files: {len(xyz_files)}")

    plot_id = 1

    for xyz_file in xyz_files:
        print(f"\n[INFO] Reading: {xyz_file}")

        is_road_xyz = "road" in xyz_file.name.lower()

        df = read_xyz(
            xyz_file,
            keep_nan=is_road_xyz,
        )

        if df is None:
            print(f"[WARN] Empty or invalid XYZ file, skip: {xyz_file}")
            continue

        # First-come first-serve numbering:
        # 01_<xyz_file_stem>.png, 02_<xyz_file_stem>.png, ...
        out_png = FIG_DIR / f"{plot_id:02d}_{xyz_file.stem}.png"
        plot_id += 1

        if is_road_xyz:
            plot_roads_xyz(
                xyz_file=xyz_file,
                df=df,
                region=region,
                out_png=out_png,
            )
        else:
            plot_xyz_points(
                xyz_file=xyz_file,
                df=df,
                region=region,
                out_png=out_png,
            )

    print("\n========== DONE ==========")
    print(f"Figures saved in: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
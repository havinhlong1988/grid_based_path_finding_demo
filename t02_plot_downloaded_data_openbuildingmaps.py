#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Hoa Lac OpenBuildingMap outputs.

Input folder:
    output_hoalac_openbuildingmap/

Expected files:
    obm_buildings_hoalac_clipped.gpkg
    obm_buildings_centroid_hoalac.xyz
    obm_buildings_vertices_hoalac.xyz
    obm_buildings_grid_hoalac.xyz
    obm_summary.csv

Output folder:
    figures/02_openbuildingmap/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt


# ============================================================
# USER INPUT PARAMETERS
# ============================================================

DATA_DIR = Path("output_hoalac_openbuildingmap")
FIG_DIR = Path("figures/01_download_xyz_openbuildingmap")

BUILDINGS_GPKG = DATA_DIR / "obm_buildings_hoalac_clipped.gpkg"
CENTROID_XYZ = DATA_DIR / "obm_buildings_centroid_hoalac.xyz"
VERTICES_XYZ = DATA_DIR / "obm_buildings_vertices_hoalac.xyz"
GRID_XYZ = DATA_DIR / "obm_buildings_grid_hoalac.xyz"
SUMMARY_CSV = DATA_DIR / "obm_summary.csv"

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

REGION_PADDING = 0.003
PROJECTION = "M15c"

# Point plotting
CENTROID_POINT_SIZE = "0.06c"
VERTEX_POINT_SIZE = "0.035c"

# Building polygon style
BUILDING_FILL_COLOR = "lightred"
BUILDING_FILL_TRANSPARENCY = 60
BUILDING_FILL = f"{BUILDING_FILL_COLOR}@{BUILDING_FILL_TRANSPARENCY}"

BUILDING_PEN = "0.25p,black@35"

# Boundary style
POLYGON_PEN = "1.5p,purple,-"

# Color map
BUILDING_CMAP = "batlow"
HEIGHT_LABEL = "Building height (m)"

# Surface/grid plotting
TMP_GRID_DIR = FIG_DIR / "tmp_grids"
GRID_SPACING = None          # None = estimate from XYZ
GRID_SEARCH_RADIUS = "0.001" # for nearneighbor, degree
MASK_OUTSIDE_POLYGON = True

# Histogram
HIST_BIN_WIDTH = 1.0


# ============================================================
# BASIC HELPERS
# ============================================================

def get_polygon_region(polygon, padding=0.003):
    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]

    west = min(lons) - padding
    east = max(lons) + padding
    south = min(lats) - padding
    north = max(lats) + padding

    return [west, east, south, north]


def make_map_title(text):
    return f'WSen+t"{text}"'


def plot_hoalac_boundary(fig):
    lons = [p[0] for p in HOALAC_POLYGON]
    lats = [p[1] for p in HOALAC_POLYGON]

    fig.plot(
        x=lons,
        y=lats,
        pen=POLYGON_PEN,
    )


def read_xyz_3col(xyz_file):
    xyz_file = Path(xyz_file)

    if not xyz_file.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_file}")

    if xyz_file.stat().st_size == 0:
        return pd.DataFrame(columns=["lon", "lat", "value"])

    df = pd.read_csv(
        xyz_file,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "value"],
        usecols=[0, 1, 2],
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["lon", "lat", "value"])

    return df


def robust_zrange(values, lower=2, upper=98):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0, 1.0

    zmin = float(np.percentile(values, lower))
    zmax = float(np.percentile(values, upper))

    if np.isclose(zmin, zmax):
        zmin = float(np.nanmin(values))
        zmax = float(np.nanmax(values))

    if np.isclose(zmin, zmax):
        zmin -= 1.0
        zmax += 1.0

    return zmin, zmax


def make_height_cpt(values):
    zmin, zmax = robust_zrange(values)

    # For building height, keep zero visible
    zmin = 0.0
    zmax = max(zmax, 1.0)

    step = (zmax - zmin) / 100.0
    if step <= 0 or not np.isfinite(step):
        step = 1.0

    pygmt.makecpt(
        cmap=BUILDING_CMAP,
        series=[zmin, zmax, step],
        continuous=True,
    )


# ============================================================
# GPKG VECTOR PLOTTING
# ============================================================

def plot_building_polygons(fig, buildings_gdf):
    if buildings_gdf is None or buildings_gdf.empty:
        return

    buildings = buildings_gdf.to_crs("EPSG:4326").copy()

    for geom in buildings.geometry:
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
                fill=BUILDING_FILL,
                pen=BUILDING_PEN,
            )


def plot_building_polygons_colored_by_height(fig, buildings_gdf):
    if buildings_gdf is None or buildings_gdf.empty:
        return

    buildings = buildings_gdf.to_crs("EPSG:4326").copy()

    if "height_m" not in buildings.columns:
        buildings["height_m"] = 0.0

    for _, row in buildings.iterrows():
        geom = row.geometry
        height = float(row.get("height_m", 0.0))

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
                fill=height,
                cmap=True,
                pen=BUILDING_PEN,
            )


# ============================================================
# SURFACE GRID HELPERS
# ============================================================

def estimate_xyz_spacing(df):
    xs = np.sort(df["lon"].unique())
    ys = np.sort(df["lat"].unique())

    dxs = np.diff(xs)
    dys = np.diff(ys)

    dxs = dxs[dxs > 0]
    dys = dys[dys > 0]

    if len(dxs) == 0 or len(dys) == 0:
        return "0.0001/0.0001"

    dx = float(np.median(dxs))
    dy = float(np.median(dys))

    return f"{dx}/{dy}"


def parse_spacing(spacing):
    parts = str(spacing).split("/")
    if len(parts) == 1:
        dx = float(parts[0])
        dy = dx
    else:
        dx = float(parts[0])
        dy = float(parts[1])
    return dx, dy


def adjust_region_to_spacing(region, spacing):
    west, east, south, north = region
    dx, dy = parse_spacing(spacing)

    nx = int(np.ceil((east - west) / dx))
    ny = int(np.ceil((north - south) / dy))

    east_adj = west + nx * dx
    north_adj = south + ny * dy

    return [west, east_adj, south, north_adj]


def mask_xyz_outside_polygon(df):
    """
    Mask outside polygon directly at XYZ level using shapely/geopandas.
    """
    points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    poly = gpd.GeoDataFrame(
        {"name": ["Hoa_Lac"]},
        geometry=[gpd.GeoSeries.from_wkt([None]).iloc[0] if False else None],
        crs="EPSG:4326",
    )

    from shapely.geometry import Polygon
    poly = gpd.GeoDataFrame(
        {"name": ["Hoa_Lac"]},
        geometry=[Polygon(HOALAC_POLYGON)],
        crs="EPSG:4326",
    )

    inside = gpd.sjoin(
        points,
        poly,
        how="inner",
        predicate="within",
    )

    return inside.drop(columns=["geometry", "index_right", "name"], errors="ignore")


def make_grid_from_obm_grid_xyz(df, region, out_grid):
    """
    Convert OBM building grid XYZ to GMT grid.

    Use nearneighbor rather than surface, because building grid has sharp
    footprint/no-footprint boundaries.
    """
    out_grid = Path(out_grid)
    out_grid.parent.mkdir(parents=True, exist_ok=True)

    if GRID_SPACING is None:
        spacing = estimate_xyz_spacing(df)
    else:
        spacing = GRID_SPACING

    grid_region = adjust_region_to_spacing(region, spacing)

    sub = df[
        (df["lon"] >= grid_region[0]) &
        (df["lon"] <= grid_region[1]) &
        (df["lat"] >= grid_region[2]) &
        (df["lat"] <= grid_region[3])
    ].copy()

    if MASK_OUTSIDE_POLYGON:
        sub = mask_xyz_outside_polygon(sub)

    if sub.empty:
        raise ValueError("No OBM grid points inside polygon/region.")

    print(f"[INFO] OBM grid spacing: {spacing}")
    print(f"[INFO] OBM grid region: {grid_region}")

    pygmt.nearneighbor(
        data=sub[["lon", "lat", "value"]],
        region=grid_region,
        spacing=spacing,
        search_radius=GRID_SEARCH_RADIUS,
        outgrid=str(out_grid),
    )

    return out_grid


# ============================================================
# PLOT FUNCTIONS
# ============================================================

def plot_obm_overview(region):
    out_png = FIG_DIR / "00_obm_buildings_overview.png"

    print("\n[INFO] Plot OBM overview")

    if not BUILDINGS_GPKG.exists():
        raise FileNotFoundError(f"Building GPKG not found: {BUILDINGS_GPKG}")

    buildings = gpd.read_file(BUILDINGS_GPKG)

    fig = pygmt.Figure()

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            make_map_title("OpenBuildingMap buildings: Hoa Lac Hi-Tech Park"),
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )

    plot_building_polygons(
        fig=fig,
        buildings_gdf=buildings,
    )

    plot_hoalac_boundary(fig)

    fig.plot(
        x=[-1000],
        y=[-1000],
        style="s0.25c",
        fill=BUILDING_FILL,
        pen=BUILDING_PEN,
        label="OBM building footprint",
    )

    fig.plot(
        x=[-1000],
        y=[-1000],
        pen=POLYGON_PEN,
        label="Hoa Lac boundary",
    )

    fig.legend(
        position="JBL+jBL+o0.2c/0.2c",
        box="+gwhite@10+p0.5p,black",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"[OK] Saved: {out_png}")


def plot_obm_building_grid(region):
    out_png = FIG_DIR / "01_obm_buildings_grid.png"
    out_grid = TMP_GRID_DIR / "obm_buildings_grid.nc"

    print("\n[INFO] Plot OBM building grid")

    df = read_xyz_3col(GRID_XYZ)

    if df.empty:
        print(f"[WARN] Empty grid XYZ: {GRID_XYZ}")
        return

    make_height_cpt(df["value"].to_numpy())

    grid_file = make_grid_from_obm_grid_xyz(
        df=df,
        region=region,
        out_grid=out_grid,
    )

    fig = pygmt.Figure()

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            make_map_title("OpenBuildingMap building height grid"),
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )

    fig.grdimage(
        grid=str(grid_file),
        cmap=True,
        nan_transparent=True,
    )

    plot_hoalac_boundary(fig)

    fig.colorbar(
        frame=f'af+l"{HEIGHT_LABEL}"',
        position="JBC+w10c/0.35c+h+o0c/0.8c",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"[OK] Saved: {out_png}")


def plot_obm_centroids(region):
    out_png = FIG_DIR / "02_obm_buildings_centroid.png"

    print("\n[INFO] Plot OBM building centroids")

    df = read_xyz_3col(CENTROID_XYZ)

    if df.empty:
        print(f"[WARN] Empty centroid XYZ: {CENTROID_XYZ}")
        return

    make_height_cpt(df["value"].to_numpy())

    fig = pygmt.Figure()

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            make_map_title("OpenBuildingMap building centroids"),
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )

    fig.plot(
        x=df["lon"],
        y=df["lat"],
        style=f"c{CENTROID_POINT_SIZE}",
        fill=df["value"],
        cmap=True,
        pen="0.1p,black@40",
    )

    plot_hoalac_boundary(fig)

    fig.colorbar(
        frame=f'af+l"{HEIGHT_LABEL}"',
        position="JBC+w10c/0.35c+h+o0c/0.8c",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"[OK] Saved: {out_png}")


def plot_obm_vertices(region):
    out_png = FIG_DIR / "03_obm_buildings_vertices.png"

    print("\n[INFO] Plot OBM building vertices")

    df = read_xyz_3col(VERTICES_XYZ)

    if df.empty:
        print(f"[WARN] Empty vertices XYZ: {VERTICES_XYZ}")
        return

    make_height_cpt(df["value"].to_numpy())

    fig = pygmt.Figure()

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            make_map_title("OpenBuildingMap building vertices"),
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )

    fig.plot(
        x=df["lon"],
        y=df["lat"],
        style=f"c{VERTEX_POINT_SIZE}",
        fill=df["value"],
        cmap=True,
        pen=None,
    )

    plot_hoalac_boundary(fig)

    fig.colorbar(
        frame=f'af+l"{HEIGHT_LABEL}"',
        position="JBC+w10c/0.35c+h+o0c/0.8c",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"[OK] Saved: {out_png}")


def plot_height_histogram():
    out_png = FIG_DIR / "04_obm_building_height_histogram.png"

    print("\n[INFO] Plot OBM building height histogram")

    if not BUILDINGS_GPKG.exists():
        raise FileNotFoundError(f"Building GPKG not found: {BUILDINGS_GPKG}")

    buildings = gpd.read_file(BUILDINGS_GPKG)

    if "height_m" not in buildings.columns:
        print("[WARN] height_m column not found. Skip histogram.")
        return

    heights = buildings["height_m"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()

    if heights.empty:
        print("[WARN] No valid height values. Skip histogram.")
        return

    hmin = 0.0
    hmax = max(float(np.ceil(heights.max())), 1.0)
    bins = np.arange(hmin, hmax + HIST_BIN_WIDTH, HIST_BIN_WIDTH)

    counts, edges = np.histogram(heights.to_numpy(), bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hist_df = pd.DataFrame(
        {
            "height": centers,
            "count": counts,
        }
    )

    fig = pygmt.Figure()

    fig.basemap(
        region=[0, hmax, 0, max(counts.max() * 1.1, 1)],
        projection="X15c/8c",
        frame=[
            'WSen+t"OpenBuildingMap building height distribution"',
            "xaf+lBuilding height (m)",
            "yaf+lCount",
        ],
    )

    fig.plot(
        x=hist_df["height"],
        y=hist_df["count"],
        style=f"b{HIST_BIN_WIDTH}u",
        fill="gray70",
        pen="0.3p,black",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"[OK] Saved: {out_png}")


# ============================================================
# MAIN
# ============================================================

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TMP_GRID_DIR.mkdir(parents=True, exist_ok=True)

    region = get_polygon_region(
        HOALAC_POLYGON,
        padding=REGION_PADDING,
    )

    print("\n========== PLOT HOA LAC OPENBUILDINGMAP DATA ==========")
    print(f"[INFO] Input folder:  {DATA_DIR}")
    print(f"[INFO] Output folder: {FIG_DIR}")
    print(f"[INFO] Region:        {region}")

    if SUMMARY_CSV.exists():
        print(f"[INFO] Summary file: {SUMMARY_CSV}")
        try:
            summary = pd.read_csv(SUMMARY_CSV)
            print(summary.head(20).to_string(index=False))
        except Exception as e:
            print(f"[WARN] Could not read summary CSV: {e}")

    plot_obm_overview(region)
    plot_obm_building_grid(region)
    plot_obm_centroids(region)
    plot_obm_vertices(region)
    plot_height_histogram()

    print("\n========== DONE ==========")
    print(f"Figures saved in: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
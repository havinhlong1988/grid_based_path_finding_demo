#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create final mixed model for scenario 1.

Main logic:
    - Read raw model:
          output/02_senario1_no_velocity/raw.xyz

      Expected format:
          lon lat elevation slowness

    - Read polygon building density grid:
          output/02_senario1_no_velocity/building_density_polygon_grid.xyz

      Expected format:
          lon lat density

    - Read OBM building polygons:
          output/01_HoaLac_studies_area/openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg

    - Select high-rise buildings from OBM polygons.
    - Only model nodes inside high-rise building footprints are changed to no-fly slowness.
    - Existing no-fly zones in raw.xyz remain unchanged.

Outputs:
    output/02_senario1_no_velocity/mixed_model.xyz
    output/02_senario1_no_velocity/mixed_model.vtk
    output/02_senario1_no_velocity/mixed_model_nodes.vtk
    output/02_senario1_no_velocity/mixed_model_cage.vtk

Figures:
    output/02_senario1_no_velocity/figures/mixed_model_overlay_on_building_density.png
    output/02_senario1_no_velocity/figures/mixed_model_2d_topview.png
    output/02_senario1_no_velocity/figures/mixed_model_3d_view_0_100m.png
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt
import xarray as xr

from shapely.geometry import Polygon


# ============================================================
# USER SETTINGS
# ============================================================

RAW_MODEL_FILE = Path("output/02_senario1_no_velocity/raw.xyz")

BUILDING_DENSITY_POLYGON_GRID = Path(
    "output/02_senario1_no_velocity/building_density_polygon_grid.xyz"
)

OBM_BUILDINGS_GPKG = Path(
    "output/01_HoaLac_studies_area/openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg"
)

OUT_DIR = Path("output/02_senario1_no_velocity")
FIG_DIR = OUT_DIR / "figures"

OUT_MIXED_XYZ = OUT_DIR / "mixed_model.xyz"
OUT_MIXED_VTK = OUT_DIR / "mixed_model.vtk"
OUT_MIXED_NODES_VTK = OUT_DIR / "mixed_model_nodes.vtk"
OUT_MIXED_CAGE_VTK = OUT_DIR / "mixed_model_cage.vtk"

OUT_OVERLAY_FIG = FIG_DIR / "mixed_model_overlay_on_building_density.png"
OUT_2D_MODEL_FIG = FIG_DIR / "mixed_model_2d_topview.png"
OUT_3D_MODEL_FIG = FIG_DIR / "mixed_model_3d_view_0_100m.png"

PROJECTION = "M15c"
DPI = 300

# Hoa Lac polygon, lon/lat.
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

# Slowness value used for no-fly zone.
# Keep same convention as raw model.
NO_FLY_SLOWNESS = 1.0e6

# High-rise selection.
HEIGHT_COLUMN = "height_m"

# If None, use percentile.
HIGHRISE_HEIGHT_M = None
HIGHRISE_HEIGHT_PERCENTILE = 90

# If True, every z node under high-rise footprint is set no-fly.
# If False, only nodes from MIN_HIGHRISE_Z to MAX_HIGHRISE_Z are set no-fly.
SET_FULL_VERTICAL_COLUMN_NO_FLY = True

MIN_HIGHRISE_Z = 0.0
MAX_HIGHRISE_Z = 3000.0

# 2D / 3D plotting.
PLOT_2D_AND_3D_MODEL = True

# 3D figure only plots this vertical range.
# This is only for visualization; the exported mixed_model still includes all z levels.
PLOT_3D_Z_MIN = 0.0
PLOT_3D_Z_MAX = 100.0

# Downsample normal 3D nodes to keep PyGMT fast.
MAX_NORMAL_3D_POINTS = 120_000
MAX_NOFLY_3D_POINTS = 200_000

PERSPECTIVE_3D = [135, 28]
ZSIZE_3D = "5c"

# Plot style.
POLYGON_PEN = "1.4p,purple"
HIGHRISE_FILL = "green@20"
HIGHRISE_PEN = "0.45p,green"
MODEL_NODE_STYLE = "c0.006c"
NOFLY_NODE_STYLE = "c0.015c"

CLEANUP_CPT_AND_TEMP_FILES = True


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def get_hoalac_polygon():
    geom = Polygon(HOALAC_POLYGON)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def get_aoi_gdf():
    return gpd.GeoDataFrame(
        {"name": ["HoaLac_polygon"]},
        geometry=[get_hoalac_polygon()],
        crs="EPSG:4326",
    )


def polygon_to_dataframe():
    return pd.DataFrame(HOALAC_POLYGON, columns=["x", "y"])


def get_region_from_polygon(padding=REGION_PADDING):
    poly_df = polygon_to_dataframe()

    xmin = float(poly_df["x"].min()) - padding
    xmax = float(poly_df["x"].max()) + padding
    ymin = float(poly_df["y"].min()) - padding
    ymax = float(poly_df["y"].max()) + padding

    return [xmin, xmax, ymin, ymax]


def start_map(region, title):
    fig = pygmt.Figure()

    pygmt.config(
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd:mmF",
        FONT_LABEL="10p",
        FONT_ANNOT_PRIMARY="9p",
    )

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            f'WSne+t"{title}"',
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )

    return fig


def plot_aoi_boundary(fig):
    poly_df = polygon_to_dataframe()

    fig.plot(
        x=poly_df["x"],
        y=poly_df["y"],
        pen=POLYGON_PEN,
        fill=None,
        label="Hoa Lac boundary",
    )


def safe_polygons(geom):
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "Polygon":
        return [geom]

    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)

    return []


def plot_polygons_constant(fig, gdf, fill=None, pen="0.2p,black", label=None):
    if gdf is None or gdf.empty:
        return

    first = True

    for geom in gdf.geometry:
        for poly in safe_polygons(geom):
            x, y = poly.exterior.xy

            kwargs = {
                "x": list(x),
                "y": list(y),
                "fill": fill,
                "pen": pen,
            }

            if label is not None and first:
                kwargs["label"] = label
                first = False

            fig.plot(**kwargs)


def plot_polygon_3d_at_z(fig, gdf, z_value, pen, fill=None, label=None):
    if gdf is None or gdf.empty:
        return

    first = True

    for geom in gdf.geometry:
        for poly in safe_polygons(geom):
            x, y = poly.exterior.xy
            x = list(x)
            y = list(y)
            z = [z_value] * len(x)

            kwargs = {
                "x": x,
                "y": y,
                "z": z,
                "pen": pen,
                "perspective": PERSPECTIVE_3D,
            }

            if fill is not None:
                kwargs["fill"] = fill

            if label is not None and first:
                kwargs["label"] = label
                first = False

            fig.plot3d(**kwargs)


# ============================================================
# READ INPUTS
# ============================================================

def read_raw_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Raw model file not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )

    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 4:
        raise ValueError(
            f"raw.xyz must have at least 4 columns: lon lat z slowness. File: {path}"
        )

    df = df.iloc[:, :4].copy()
    df.columns = ["x", "y", "z", "slowness"]

    for col in ["x", "y", "z", "slowness"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["x", "y", "z", "slowness"]).copy()

    print("========== RAW MODEL ==========")
    print(f"Input raw model: {path}")
    print(f"Nodes:           {len(df):,}")
    print(f"x range:         {df['x'].min()} -> {df['x'].max()}")
    print(f"y range:         {df['y'].min()} -> {df['y'].max()}")
    print(f"z range:         {df['z'].min()} -> {df['z'].max()}")
    print(f"slowness range:  {df['slowness'].min()} -> {df['slowness'].max()}")

    return df


def read_density_grid_xyz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Building density grid not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )

    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 3:
        raise ValueError(
            f"Density grid must have 3 columns: lon lat density. File: {path}"
        )

    df = df.iloc[:, :3].copy()
    df.columns = ["x", "y", "density"]

    for col in ["x", "y", "density"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["x", "y", "density"]).copy()

    print("")
    print("========== BUILDING DENSITY GRID ==========")
    print(f"Input density grid: {path}")
    print(f"Grid nodes:         {len(df):,}")
    print(f"density range:      {df['density'].min()} -> {df['density'].max()}")

    return df


def density_xyz_to_xarray(density_df):
    xs = np.sort(density_df["x"].unique())
    ys = np.sort(density_df["y"].unique())

    pivot = density_df.pivot_table(
        index="y",
        columns="x",
        values="density",
        aggfunc="mean",
    )

    pivot = pivot.reindex(index=ys, columns=xs)

    grid = xr.DataArray(
        pivot.to_numpy(),
        coords={
            "lat": ys,
            "lon": xs,
        },
        dims=("lat", "lon"),
        name="building_density",
    )

    return grid


def load_obm_buildings(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"OBM building polygon file not found: {path}")

    print("")
    print("========== LOAD OBM BUILDINGS ==========")
    print(f"OBM polygon file: {path}")

    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"OBM GPKG is empty: {path}")

    if gdf.crs is None:
        print("[WARNING] OBM file has no CRS. Assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    aoi = get_aoi_gdf()

    n_before = len(gdf)

    try:
        gdf = gpd.clip(gdf, aoi).copy()
    except Exception as exc:
        print(f"[WARNING] gpd.clip failed, using intersects only: {exc}")
        gdf = gdf[gdf.intersects(aoi.geometry.iloc[0])].copy()

    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()

    print(f"Buildings before clip: {n_before:,}")
    print(f"Buildings after clip:  {len(gdf):,}")

    if HEIGHT_COLUMN in gdf.columns:
        gdf[HEIGHT_COLUMN] = pd.to_numeric(gdf[HEIGHT_COLUMN], errors="coerce")
        print(f"Height column:         {HEIGHT_COLUMN}")
        print(f"Height valid count:    {gdf[HEIGHT_COLUMN].notna().sum():,}")
        print(f"Height max:            {gdf[HEIGHT_COLUMN].max()}")
    else:
        print(f"[WARNING] Height column not found: {HEIGHT_COLUMN}")

    return gdf


def select_highrise_buildings(buildings):
    if buildings is None or buildings.empty:
        raise ValueError("No building polygons available for high-rise selection.")

    gdf = buildings.copy()

    if HEIGHT_COLUMN in gdf.columns:
        vals = pd.to_numeric(gdf[HEIGHT_COLUMN], errors="coerce")

        if vals.notna().sum() > 0 and vals.max() > 0:
            gdf["_height_for_model"] = vals.fillna(0.0)

            if HIGHRISE_HEIGHT_M is not None:
                threshold = float(HIGHRISE_HEIGHT_M)
            else:
                positive = gdf.loc[gdf["_height_for_model"] > 0, "_height_for_model"]
                threshold = float(np.nanpercentile(positive, HIGHRISE_HEIGHT_PERCENTILE))

            highrise = gdf[gdf["_height_for_model"] >= threshold].copy()
            method = f"{HEIGHT_COLUMN} >= {threshold:.2f} m"

            print("")
            print("========== HIGH-RISE SELECTION ==========")
            print(f"Method:          {method}")
            print(f"All buildings:   {len(gdf):,}")
            print(f"High-rise count: {len(highrise):,}")

            if highrise.empty:
                raise ValueError("High-rise selection is empty. Lower threshold.")

            return highrise, method

    # Fallback: select largest footprints by area.
    centroid = gdf.geometry.unary_union.centroid
    zone = int(np.floor((centroid.x + 180.0) / 6.0) + 1)
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone

    gdf_m = gdf.to_crs(epsg)
    areas = gdf_m.geometry.area

    threshold = float(np.nanpercentile(areas, HIGHRISE_HEIGHT_PERCENTILE))
    highrise = gdf.loc[areas >= threshold].copy()

    method = f"footprint area >= P{HIGHRISE_HEIGHT_PERCENTILE}"

    print("")
    print("========== HIGH-RISE SELECTION ==========")
    print(f"Method:          {method}")
    print(f"All buildings:   {len(gdf):,}")
    print(f"High-rise count: {len(highrise):,}")

    if highrise.empty:
        raise ValueError("High-rise footprint selection is empty.")

    return highrise, method


# ============================================================
# MIXED MODEL
# ============================================================

def find_nodes_inside_highrise(raw_df, highrise_gdf):
    """
    Find raw model nodes whose horizontal location is inside high-rise buildings.

    The flag is assigned by unique x/y first, then merged back to all z layers.
    """
    xy_df = raw_df[["x", "y"]].drop_duplicates().reset_index(drop=True).copy()
    xy_df["xy_id"] = xy_df.index

    points_gdf = gpd.GeoDataFrame(
        xy_df,
        geometry=gpd.points_from_xy(xy_df["x"], xy_df["y"]),
        crs="EPSG:4326",
    )

    highrise = highrise_gdf.to_crs("EPSG:4326").copy()
    highrise = highrise[highrise.geometry.notna() & (~highrise.geometry.is_empty)].copy()

    joined = gpd.sjoin(
        points_gdf,
        highrise[["geometry"]],
        how="left",
        predicate="within",
    )

    inside_ids = joined.loc[joined["index_right"].notna(), "xy_id"].unique()

    xy_df["inside_highrise"] = False
    xy_df.loc[xy_df["xy_id"].isin(inside_ids), "inside_highrise"] = True

    raw_out = raw_df.merge(
        xy_df[["x", "y", "inside_highrise"]],
        on=["x", "y"],
        how="left",
    )

    raw_out["inside_highrise"] = raw_out["inside_highrise"].fillna(False)

    if SET_FULL_VERTICAL_COLUMN_NO_FLY:
        z_mask = np.ones(len(raw_out), dtype=bool)
    else:
        z_mask = (
            (raw_out["z"].to_numpy() >= MIN_HIGHRISE_Z)
            & (raw_out["z"].to_numpy() <= MAX_HIGHRISE_Z)
        )

    highrise_node_mask = raw_out["inside_highrise"].to_numpy() & z_mask

    print("")
    print("========== HIGH-RISE NODE OVERLAY ==========")
    print(f"Unique xy nodes:             {len(xy_df):,}")
    print(f"xy inside high-rise:         {int(xy_df['inside_highrise'].sum()):,}")
    print(f"3D nodes inside high-rise:   {int(highrise_node_mask.sum()):,}")
    print(f"Full vertical column no-fly: {SET_FULL_VERTICAL_COLUMN_NO_FLY}")

    return raw_out, highrise_node_mask


def create_mixed_model(raw_df, highrise_gdf):
    raw_with_flags, highrise_node_mask = find_nodes_inside_highrise(
        raw_df=raw_df,
        highrise_gdf=highrise_gdf,
    )

    mixed = raw_with_flags.copy()

    mixed["original_slowness"] = mixed["slowness"].copy()
    mixed["highrise_nofly"] = highrise_node_mask.astype(int)

    # Only set high-rise regions to no-fly.
    # Existing no-fly zones remain untouched because other nodes are not modified.
    mixed.loc[highrise_node_mask, "slowness"] = NO_FLY_SLOWNESS

    changed = mixed["slowness"] != mixed["original_slowness"]

    print("")
    print("========== MIXED MODEL RESULT ==========")
    print(f"Total nodes:                   {len(mixed):,}")
    print(f"Nodes changed to high-rise NF: {int(changed.sum()):,}")
    print(f"Existing no-fly nodes kept:    {int((raw_df['slowness'] >= NO_FLY_SLOWNESS).sum()):,}")
    print(f"Final no-fly nodes:            {int((mixed['slowness'] >= NO_FLY_SLOWNESS).sum()):,}")
    print(f"Final slowness range:          {mixed['slowness'].min()} -> {mixed['slowness'].max()}")

    return mixed


def save_mixed_xyz(mixed_df, out_file):
    out_df = mixed_df[["x", "y", "z", "slowness"]].copy()

    out_df.to_csv(
        out_file,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    print(f"[OK] Saved mixed model XYZ: {out_file}")


# ============================================================
# VTK EXPORT
# ============================================================

def write_legacy_structured_grid_vtk(df, out_file, scalar_col="slowness"):
    """
    Write full mixed model as legacy ASCII STRUCTURED_GRID VTK.

    Assumes regular grid with unique x, y, z.
    If grid is incomplete, it falls back to POLYDATA nodes.
    """
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    zs = np.sort(df["z"].unique())

    nx, ny, nz = len(xs), len(ys), len(zs)
    expected = nx * ny * nz

    if expected != len(df):
        print(
            "[WARNING] Raw model is not a complete structured grid. "
            "Writing mixed_model.vtk as POLYDATA instead."
        )
        write_legacy_polydata_nodes_vtk(
            df=df,
            out_file=out_file,
            scalar_col=scalar_col,
        )
        return

    lookup = df.set_index(["x", "y", "z"])[scalar_col].to_dict()

    points = []
    scalars = []

    # VTK structured order: x fastest, then y, then z.
    for z in zs:
        for y in ys:
            for x in xs:
                points.append((x, y, z))
                scalars.append(float(lookup[(x, y, z)]))

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("mixed_model_structured_grid\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"POINTS {len(points)} float\n")

        for x, y, z in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")

        f.write(f"\nPOINT_DATA {len(points)}\n")
        f.write(f"SCALARS {scalar_col} float 1\n")
        f.write("LOOKUP_TABLE default\n")

        for v in scalars:
            f.write(f"{v:.8f}\n")

    print(f"[OK] Saved structured-grid VTK: {out_file}")


def write_legacy_polydata_nodes_vtk(df, out_file, scalar_col="slowness"):
    """
    Write model nodes as legacy ASCII POLYDATA VTK.
    """
    n = len(df)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("mixed_model_nodes\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")

        for row in df.itertuples(index=False):
            f.write(f"{row.x:.8f} {row.y:.8f} {row.z:.8f}\n")

        f.write(f"\nVERTICES {n} {2 * n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")

        f.write(f"\nPOINT_DATA {n}\n")
        f.write(f"SCALARS {scalar_col} float 1\n")
        f.write("LOOKUP_TABLE default\n")

        for v in df[scalar_col].to_numpy():
            f.write(f"{float(v):.8f}\n")

        if "highrise_nofly" in df.columns:
            f.write("\nSCALARS highrise_nofly int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for v in df["highrise_nofly"].to_numpy():
                f.write(f"{int(v)}\n")

    print(f"[OK] Saved node POLYDATA VTK: {out_file}")


def write_model_cage_vtk(df, out_file):
    """
    Write rectangular model cage from raw model extent as POLYDATA lines.
    """
    xmin, xmax = float(df["x"].min()), float(df["x"].max())
    ymin, ymax = float(df["y"].min()), float(df["y"].max())
    zmin, zmax = float(df["z"].min()), float(df["z"].max())

    points = [
        (xmin, ymin, zmin),
        (xmax, ymin, zmin),
        (xmax, ymax, zmin),
        (xmin, ymax, zmin),
        (xmin, ymin, zmax),
        (xmax, ymin, zmax),
        (xmax, ymax, zmax),
        (xmin, ymax, zmax),
    ]

    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("mixed_model_cage\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")

        for x, y, z in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")

        f.write(f"\nLINES {len(lines)} {len(lines) * 3}\n")

        for i, j in lines:
            f.write(f"2 {i} {j}\n")

    print(f"[OK] Saved model cage VTK: {out_file}")


# ============================================================
# 2D / 3D MODEL PLOTS
# ============================================================

def get_topview_nodes(mixed_df):
    """
    Build 2D top-view model nodes.

    If any vertical node at one x/y has no-fly slowness,
    mark the horizontal x/y location as no-fly.
    """
    df = mixed_df.copy()

    df["is_nofly"] = df["slowness"] >= NO_FLY_SLOWNESS

    top = (
        df.groupby(["x", "y"], as_index=False)
        .agg(
            max_slowness=("slowness", "max"),
            min_slowness=("slowness", "min"),
            is_nofly=("is_nofly", "max"),
            highrise_nofly=("highrise_nofly", "max"),
        )
    )

    return top


def plot_overlay_check(density_grid, mixed_df, highrise_gdf, region, out_png):
    print("")
    print("========== PLOT MIXED MODEL OVERLAY ==========")

    cpt_file = OUT_DIR / "mixed_model_overlay_density.cpt"

    pygmt.makecpt(
        cmap="hot",
        series=[0, 1, 0.05],
        reverse=True,
        continuous=True,
        output=str(cpt_file),
    )

    fig = start_map(region, "Mixed model nodes over building density")

    fig.grdimage(
        grid=density_grid,
        cmap=str(cpt_file),
        transparency=0,
    )

    node_xy = mixed_df[["x", "y", "highrise_nofly"]].drop_duplicates().copy()

    normal_xy = node_xy[node_xy["highrise_nofly"] == 0]
    nofly_xy = node_xy[node_xy["highrise_nofly"] == 1]

    fig.plot(
        x=normal_xy["x"],
        y=normal_xy["y"],
        style=MODEL_NODE_STYLE,
        fill="black",
        pen=None,
        transparency=75,
        label="Model node",
    )

    if not nofly_xy.empty:
        fig.plot(
            x=nofly_xy["x"],
            y=nofly_xy["y"],
            style=NOFLY_NODE_STYLE,
            fill="cyan",
            pen="0.1p,black",
            transparency=5,
            label="High-rise no-fly node",
        )

    plot_polygons_constant(
        fig,
        highrise_gdf,
        fill=HIGHRISE_FILL,
        pen=HIGHRISE_PEN,
        label="High-rise building",
    )

    plot_aoi_boundary(fig)

    fig.colorbar(
        cmap=str(cpt_file),
        position="JBC+w9c/0.35c+o0c/1.0c+h",
        frame=[
            "xaf+lBuilding density",
            "y+lNormalized",
        ],
    )

    fig.legend(
        position="JBL+jBL+o0.2c/0.2c",
        box="+gwhite@10+p0.5p,black",
    )

    fig.savefig(str(out_png), dpi=DPI)

    print(f"[OK] Saved overlay check figure: {out_png}")


def plot_2d_model_topview(
    density_grid,
    mixed_df,
    highrise_gdf,
    region,
    out_png,
):
    """
    Plot final mixed model in 2D top view.
    """
    print("")
    print("========== PLOT 2D MIXED MODEL TOP VIEW ==========")

    cpt_file = OUT_DIR / "mixed_model_2d_density.cpt"

    pygmt.makecpt(
        cmap="hot",
        series=[0, 1, 0.05],
        reverse=True,
        continuous=True,
        output=str(cpt_file),
    )

    top = get_topview_nodes(mixed_df)

    normal = top[top["is_nofly"] == False].copy()
    nofly = top[top["is_nofly"] == True].copy()
    highrise_nofly = top[top["highrise_nofly"] == 1].copy()

    fig = start_map(region, "Final mixed model: 2D top view")

    fig.grdimage(
        grid=density_grid,
        cmap=str(cpt_file),
        transparency=0,
    )

    if not normal.empty:
        fig.plot(
            x=normal["x"],
            y=normal["y"],
            style="c0.006c",
            fill="black",
            pen=None,
            transparency=80,
            label="Model node",
        )

    if not nofly.empty:
        fig.plot(
            x=nofly["x"],
            y=nofly["y"],
            style="c0.012c",
            fill="gray35",
            pen=None,
            transparency=30,
            label="Final no-fly node",
        )

    if not highrise_nofly.empty:
        fig.plot(
            x=highrise_nofly["x"],
            y=highrise_nofly["y"],
            style="c0.018c",
            fill="cyan",
            pen="0.1p,black",
            transparency=5,
            label="High-rise no-fly node",
        )

    plot_polygons_constant(
        fig,
        highrise_gdf,
        fill=HIGHRISE_FILL,
        pen=HIGHRISE_PEN,
        label="High-rise building",
    )

    plot_aoi_boundary(fig)

    fig.colorbar(
        cmap=str(cpt_file),
        position="JBC+w9c/0.35c+o0c/1.0c+h",
        frame=[
            "xaf+lBuilding density",
            "y+lNormalized",
        ],
    )

    fig.legend(
        position="JBL+jBL+o0.2c/0.2c",
        box="+gwhite@10+p0.5p,black",
    )

    fig.savefig(str(out_png), dpi=DPI)

    print(f"[OK] Saved 2D mixed model figure: {out_png}")


def sample_for_3d_plot(df, max_points, random_state=12345):
    if df is None or df.empty:
        return df

    if max_points is None:
        return df

    if len(df) <= max_points:
        return df

    return df.sample(
        n=max_points,
        random_state=random_state,
    ).copy()


def plot_3d_model_view_0_100m(
    mixed_df,
    highrise_gdf,
    region,
    out_png,
):
    """
    Plot final mixed model in 3D, limited to 0-100 m.

    This only limits the figure, not the exported model.
    """
    print("")
    print("========== PLOT 3D MIXED MODEL 0-100 M ==========")

    df = mixed_df.copy()

    df = df[
        (df["z"] >= PLOT_3D_Z_MIN)
        & (df["z"] <= PLOT_3D_Z_MAX)
    ].copy()

    if df.empty:
        raise ValueError(
            f"No model nodes found between z={PLOT_3D_Z_MIN} and z={PLOT_3D_Z_MAX}."
        )

    df["is_nofly"] = df["slowness"] >= NO_FLY_SLOWNESS

    normal = df[df["is_nofly"] == False].copy()
    nofly = df[df["is_nofly"] == True].copy()
    highrise_nofly = df[df["highrise_nofly"] == 1].copy()

    normal_plot = sample_for_3d_plot(
        normal,
        MAX_NORMAL_3D_POINTS,
        random_state=12345,
    )

    nofly_plot = sample_for_3d_plot(
        nofly,
        MAX_NOFLY_3D_POINTS,
        random_state=12346,
    )

    highrise_plot = sample_for_3d_plot(
        highrise_nofly,
        MAX_NOFLY_3D_POINTS,
        random_state=12347,
    )

    xmin, xmax, ymin, ymax = region
    zmin = PLOT_3D_Z_MIN
    zmax = PLOT_3D_Z_MAX

    region3d = [xmin, xmax, ymin, ymax, zmin, zmax]

    print(f"3D plot z range:          {zmin} -> {zmax}")
    print(f"3D normal nodes plotted:  {len(normal_plot):,}")
    print(f"3D no-fly nodes plotted:  {len(nofly_plot):,}")
    print(f"3D high-rise NF plotted:  {len(highrise_plot):,}")

    fig = pygmt.Figure()

    pygmt.config(
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd:mmF",
        FONT_LABEL="10p",
        FONT_ANNOT_PRIMARY="8p",
    )

    fig.basemap(
        region=region3d,
        projection=PROJECTION,
        zsize=ZSIZE_3D,
        perspective=PERSPECTIVE_3D,
        frame=[
            'WSneZ+t"Final mixed model: 3D view 0-100 m"',
            "xaf+lLongitude",
            "yaf+lLatitude",
            "zaf+lElevation (m)",
        ],
    )

    if normal_plot is not None and not normal_plot.empty:
        fig.plot3d(
            x=normal_plot["x"],
            y=normal_plot["y"],
            z=normal_plot["z"],
            style="c0.015c",
            fill="black",
            pen=None,
            transparency=88,
            perspective=PERSPECTIVE_3D,
            label="Model node",
        )

    if nofly_plot is not None and not nofly_plot.empty:
        fig.plot3d(
            x=nofly_plot["x"],
            y=nofly_plot["y"],
            z=nofly_plot["z"],
            style="c0.022c",
            fill="gray35",
            pen=None,
            transparency=45,
            perspective=PERSPECTIVE_3D,
            label="Final no-fly node",
        )

    if highrise_plot is not None and not highrise_plot.empty:
        fig.plot3d(
            x=highrise_plot["x"],
            y=highrise_plot["y"],
            z=highrise_plot["z"],
            style="c0.035c",
            fill="cyan",
            pen="0.1p,black",
            transparency=10,
            perspective=PERSPECTIVE_3D,
            label="High-rise no-fly node",
        )

    # High-rise footprints at zmin and zmax.
    plot_polygon_3d_at_z(
        fig,
        highrise_gdf,
        z_value=zmin,
        pen=HIGHRISE_PEN,
        fill=HIGHRISE_FILL,
        label="High-rise footprint",
    )

    plot_polygon_3d_at_z(
        fig,
        highrise_gdf,
        z_value=zmax,
        pen="0.35p,green",
        fill=None,
        label=None,
    )

    # Hoa Lac polygon at bottom.
    poly_df = polygon_to_dataframe()
    fig.plot3d(
        x=poly_df["x"],
        y=poly_df["y"],
        z=[zmin] * len(poly_df),
        pen=POLYGON_PEN,
        perspective=PERSPECTIVE_3D,
        label="Hoa Lac boundary",
    )

    fig.legend(
        position="JBL+jBL+o0.2c/0.2c",
        box="+gwhite@10+p0.5p,black",
    )

    fig.savefig(str(out_png), dpi=DPI)

    print(f"[OK] Saved 3D mixed model figure: {out_png}")


# ============================================================
# CLEANUP
# ============================================================

def cleanup_cpt_and_temp_files():
    if not CLEANUP_CPT_AND_TEMP_FILES:
        return

    print("")
    print("========== CLEANUP TEMP FILES ==========")

    removed = 0

    for folder in [OUT_DIR, FIG_DIR]:
        for path in folder.glob("*.cpt"):
            if path.is_file():
                try:
                    path.unlink()
                    removed += 1
                    print(f"[CLEAN] Removed: {path}")
                except Exception as exc:
                    print(f"[WARN] Could not remove {path}: {exc}")

    print(f"[OK] Cleanup done. Removed files: {removed}")


# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_dirs()

    print("\n========== CREATE MIXED MODEL ==========")

    region = get_region_from_polygon(padding=REGION_PADDING)

    raw_df = read_raw_model(RAW_MODEL_FILE)

    density_df = read_density_grid_xyz(BUILDING_DENSITY_POLYGON_GRID)
    density_grid = density_xyz_to_xarray(density_df)

    buildings = load_obm_buildings(OBM_BUILDINGS_GPKG)
    highrise, highrise_method = select_highrise_buildings(buildings)

    mixed_df = create_mixed_model(
        raw_df=raw_df,
        highrise_gdf=highrise,
    )

    save_mixed_xyz(
        mixed_df=mixed_df,
        out_file=OUT_MIXED_XYZ,
    )

    write_legacy_structured_grid_vtk(
        df=mixed_df,
        out_file=OUT_MIXED_VTK,
        scalar_col="slowness",
    )

    write_legacy_polydata_nodes_vtk(
        df=mixed_df,
        out_file=OUT_MIXED_NODES_VTK,
        scalar_col="slowness",
    )

    write_model_cage_vtk(
        df=mixed_df,
        out_file=OUT_MIXED_CAGE_VTK,
    )

    plot_overlay_check(
        density_grid=density_grid,
        mixed_df=mixed_df,
        highrise_gdf=highrise,
        region=region,
        out_png=OUT_OVERLAY_FIG,
    )

    if PLOT_2D_AND_3D_MODEL:
        plot_2d_model_topview(
            density_grid=density_grid,
            mixed_df=mixed_df,
            highrise_gdf=highrise,
            region=region,
            out_png=OUT_2D_MODEL_FIG,
        )

        plot_3d_model_view_0_100m(
            mixed_df=mixed_df,
            highrise_gdf=highrise,
            region=region,
            out_png=OUT_3D_MODEL_FIG,
        )

    cleanup_cpt_and_temp_files()

    print("\n========== DONE ==========")
    print(f"Mixed model XYZ:        {OUT_MIXED_XYZ}")
    print(f"Mixed model VTK:        {OUT_MIXED_VTK}")
    print(f"Mixed model nodes VTK:  {OUT_MIXED_NODES_VTK}")
    print(f"Mixed model cage VTK:   {OUT_MIXED_CAGE_VTK}")
    print(f"Overlay check figure:   {OUT_OVERLAY_FIG}")
    print(f"2D model figure:        {OUT_2D_MODEL_FIG}")
    print(f"3D model figure:        {OUT_3D_MODEL_FIG}")
    print(f"High-rise method:       {highrise_method}")
    print(f"No-fly slowness:        {NO_FLY_SLOWNESS}")


if __name__ == "__main__":
    main()
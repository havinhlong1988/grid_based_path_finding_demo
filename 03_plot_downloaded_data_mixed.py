#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create final mixed model and plot categorical 2D/3D model nodes.

Main logic:
    - Read raw model:
          output/02_senario1_no_velocity/raw.xyz

      Expected format:
          lon lat elevation_m slowness

    - Read OBM building polygons:
          output/01_HoaLac_studies_area/openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg

    - Select high-rise buildings from OBM polygons.
    - Only model nodes inside high-rise building footprints are changed to no-fly slowness.
    - Existing no-fly zones in raw.xyz remain unchanged.

Categorical classes:
    0 = flyable / normal slowness
    1 = existing no-fly from raw.xyz
    2 = new high-rise no-fly

Plots:
    output/02_senario1_no_velocity/figures/mixed_model_2d_categorical.png
    output/02_senario1_no_velocity/figures/mixed_model_3d_categorical_0_100m.png

VTK outputs:
    output/02_senario1_no_velocity/mixed_model.vtk
    output/02_senario1_no_velocity/mixed_model_nodes.vtk
    output/02_senario1_no_velocity/mixed_model_cage.vtk

VTK coordinate rule:
    - If x/y look like lon/lat:
          x = lon
          y = lat
          z = elevation_m / 1000.0
    - Else:
          x = x_m / 1000.0
          y = y_m / 1000.0
          z = z_m / 1000.0
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt

from shapely.geometry import Polygon


# ============================================================
# USER SETTINGS
# ============================================================

RAW_MODEL_FILE = Path("output/02_senario1_no_velocity/raw.xyz")

OBM_BUILDINGS_GPKG = Path(
    "output/01_HoaLac_studies_area/openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg"
)

OUT_DIR = Path("output/02_senario1_no_velocity")
FIG_DIR = OUT_DIR / "figures"

OUT_MIXED_XYZ = OUT_DIR / "mixed_model.xyz"
OUT_MIXED_VTK = OUT_DIR / "mixed_model.vtk"
OUT_MIXED_NODES_VTK = OUT_DIR / "mixed_model_nodes.vtk"
OUT_MIXED_CAGE_VTK = OUT_DIR / "mixed_model_cage.vtk"

OUT_2D_MODEL_FIG = FIG_DIR / "mixed_model_2d_categorical.png"
OUT_3D_MODEL_FIG = FIG_DIR / "mixed_model_3d_categorical_0_100m.png"

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

# Slowness no-fly value.
NO_FLY_SLOWNESS = 1.0e6

# High-rise building selection.
HEIGHT_COLUMN = "height_m"
HIGHRISE_HEIGHT_M = None
HIGHRISE_HEIGHT_PERCENTILE = 90

# If True, every z node under high-rise footprint becomes no-fly.
# If False, only MIN_HIGHRISE_Z_M to MAX_HIGHRISE_Z_M becomes no-fly.
SET_FULL_VERTICAL_COLUMN_NO_FLY = True
MIN_HIGHRISE_Z_M = 0.0
MAX_HIGHRISE_Z_M = 3000.0

# 3D plotting range in meters.
PLOT_3D_Z_MIN_M = 0.0
PLOT_3D_Z_MAX_M = 100.0

# Downsample for 3D plotting only.
MAX_3D_FLYABLE_POINTS = 120_000
MAX_3D_EXISTING_NOFLY_POINTS = 120_000
MAX_3D_HIGHRISE_NOFLY_POINTS = 200_000

PERSPECTIVE_3D = [135, 28]
ZSIZE_3D = "5c"

# Plot styles.
POLYGON_PEN = "1.4p,purple"
HIGHRISE_PEN = "0.55p,green"
HIGHRISE_FILL = "green@85"

CATEGORY_STYLES = {
    0: {
        "name": "Flyable",
        "fill": "black",
        "style_2d": "c0.006c",
        "style_3d": "c0.014c",
        "transparency_2d": 80,
        "transparency_3d": 88,
    },
    1: {
        "name": "Existing no-fly",
        "fill": "red",
        "style_2d": "c0.014c",
        "style_3d": "c0.025c",
        "transparency_2d": 35,
        "transparency_3d": 45,
    },
    2: {
        "name": "High-rise no-fly",
        "fill": "cyan",
        "style_2d": "c0.022c",
        "style_3d": "c0.040c",
        "transparency_2d": 5,
        "transparency_3d": 10,
    },
}

PLOT_HIGHRISE_FOOTPRINT_ON_2D = True
PLOT_HIGHRISE_FOOTPRINT_ON_3D = True

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


def coordinates_look_lonlat(df):
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    return (
        np.nanmin(x) >= -180
        and np.nanmax(x) <= 180
        and np.nanmin(y) >= -90
        and np.nanmax(y) <= 90
    )


def vtk_coordinate_arrays(df):
    """
    Convert coordinates for VTK.

    If x/y are lon/lat:
        keep x/y as lon/lat, convert z from m to km.
    Else:
        convert x/y/z from m to km.
    """
    is_lonlat = coordinates_look_lonlat(df)

    if is_lonlat:
        xvtk = df["x"].to_numpy(dtype=float)
        yvtk = df["y"].to_numpy(dtype=float)
        zvtk = df["z"].to_numpy(dtype=float) / 1000.0
        units = "x/y=lonlat, z=km"
    else:
        xvtk = df["x"].to_numpy(dtype=float) / 1000.0
        yvtk = df["y"].to_numpy(dtype=float) / 1000.0
        zvtk = df["z"].to_numpy(dtype=float) / 1000.0
        units = "x/y/z=km"

    return xvtk, yvtk, zvtk, units


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


def plot_polygon_3d_at_zkm(fig, gdf, z_km, pen, fill=None, label=None):
    if gdf is None or gdf.empty:
        return

    first = True

    for geom in gdf.geometry:
        for poly in safe_polygons(geom):
            x, y = poly.exterior.xy
            x = list(x)
            y = list(y)
            z = [z_km] * len(x)

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
            f"raw.xyz must have 4 columns: x y z slowness. File: {path}"
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
    print(f"z range (m):     {df['z'].min()} -> {df['z'].max()}")
    print(f"slowness range:  {df['slowness'].min()} -> {df['slowness'].max()}")
    print(f"Coordinates:     {'lon/lat' if coordinates_look_lonlat(df) else 'projected meter'}")

    return df


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
    Assign high-rise flag by unique x/y, then merge to all z layers.
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
            (raw_out["z"].to_numpy() >= MIN_HIGHRISE_Z_M)
            & (raw_out["z"].to_numpy() <= MAX_HIGHRISE_Z_M)
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
    mixed["existing_nofly"] = (mixed["original_slowness"] >= NO_FLY_SLOWNESS).astype(int)
    mixed["highrise_nofly"] = highrise_node_mask.astype(int)

    # Only set high-rise regions to no-fly.
    # Existing no-fly zones remain untouched.
    mixed.loc[highrise_node_mask, "slowness"] = NO_FLY_SLOWNESS

    # Categorical class:
    # 0 = flyable
    # 1 = existing no-fly
    # 2 = high-rise no-fly
    mixed["slowness_class"] = 0
    mixed.loc[mixed["existing_nofly"] == 1, "slowness_class"] = 1
    mixed.loc[mixed["highrise_nofly"] == 1, "slowness_class"] = 2

    changed = mixed["slowness"] != mixed["original_slowness"]

    print("")
    print("========== MIXED MODEL RESULT ==========")
    print(f"Total nodes:                   {len(mixed):,}")
    print(f"Nodes changed to high-rise NF: {int(changed.sum()):,}")
    print(f"Existing no-fly nodes kept:    {int((mixed['existing_nofly'] == 1).sum()):,}")
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

    VTK coordinate conversion:
        lon/lat model:
            x = lon
            y = lat
            z = z_m / 1000
        projected-meter model:
            x = x_m / 1000
            y = y_m / 1000
            z = z_m / 1000
    """
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    zs = np.sort(df["z"].unique())

    nx, ny, nz = len(xs), len(ys), len(zs)
    expected = nx * ny * nz

    if expected != len(df):
        print(
            "[WARNING] Model is not a complete structured grid. "
            "Writing mixed_model.vtk as POLYDATA instead."
        )
        write_legacy_polydata_nodes_vtk(
            df=df,
            out_file=out_file,
            scalar_col=scalar_col,
        )
        return

    work = df.copy()
    xvtk, yvtk, zvtk, vtk_units = vtk_coordinate_arrays(work)

    work["_xvtk"] = xvtk
    work["_yvtk"] = yvtk
    work["_zvtk"] = zvtk

    lookup = work.set_index(["x", "y", "z"])

    points = []
    slowness = []
    original_slowness = []
    existing_nofly = []
    highrise_nofly = []
    slowness_class = []

    for z in zs:
        for y in ys:
            for x in xs:
                row = lookup.loc[(x, y, z)]

                points.append((row["_xvtk"], row["_yvtk"], row["_zvtk"]))
                slowness.append(float(row["slowness"]))
                original_slowness.append(float(row["original_slowness"]))
                existing_nofly.append(int(row["existing_nofly"]))
                highrise_nofly.append(int(row["highrise_nofly"]))
                slowness_class.append(int(row["slowness_class"]))

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"mixed_model_structured_grid coordinate_units={vtk_units}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"POINTS {len(points)} float\n")

        for x, y, z in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")

        f.write(f"\nPOINT_DATA {len(points)}\n")

        f.write("SCALARS slowness float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in slowness:
            f.write(f"{v:.8f}\n")

        f.write("\nSCALARS original_slowness float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in original_slowness:
            f.write(f"{v:.8f}\n")

        f.write("\nSCALARS existing_nofly int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in existing_nofly:
            f.write(f"{v}\n")

        f.write("\nSCALARS highrise_nofly int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in highrise_nofly:
            f.write(f"{v}\n")

        f.write("\nSCALARS slowness_class int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in slowness_class:
            f.write(f"{v}\n")

    print(f"[OK] Saved structured-grid VTK: {out_file}")
    print(f"[INFO] VTK coordinate units: {vtk_units}")


def write_legacy_polydata_nodes_vtk(df, out_file, scalar_col="slowness"):
    """
    Write model nodes as legacy ASCII POLYDATA VTK.
    """
    work = df.copy()
    xvtk, yvtk, zvtk, vtk_units = vtk_coordinate_arrays(work)

    work["_xvtk"] = xvtk
    work["_yvtk"] = yvtk
    work["_zvtk"] = zvtk

    n = len(work)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"mixed_model_nodes coordinate_units={vtk_units}\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")

        for row in work.itertuples(index=False):
            f.write(f"{row._xvtk:.8f} {row._yvtk:.8f} {row._zvtk:.8f}\n")

        f.write(f"\nVERTICES {n} {2 * n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")

        scalar_columns = [
            ("slowness", "float"),
            ("original_slowness", "float"),
            ("existing_nofly", "int"),
            ("highrise_nofly", "int"),
            ("slowness_class", "int"),
        ]

        f.write(f"\nPOINT_DATA {n}\n")

        for col, vtk_type in scalar_columns:
            if col not in work.columns:
                continue

            f.write(f"SCALARS {col} {vtk_type} 1\n")
            f.write("LOOKUP_TABLE default\n")

            for v in work[col].to_numpy():
                if vtk_type == "int":
                    f.write(f"{int(v)}\n")
                else:
                    f.write(f"{float(v):.8f}\n")

            f.write("\n")

    print(f"[OK] Saved node POLYDATA VTK: {out_file}")
    print(f"[INFO] VTK coordinate units: {vtk_units}")


def write_model_cage_vtk(df, out_file):
    """
    Write rectangular model cage as POLYDATA lines.

    Cage coordinates follow same VTK coordinate conversion rule:
        lon/lat model: x/y lonlat, z km
        projected-meter model: x/y/z km
    """
    xmin, xmax = float(df["x"].min()), float(df["x"].max())
    ymin, ymax = float(df["y"].min()), float(df["y"].max())
    zmin, zmax = float(df["z"].min()), float(df["z"].max())

    cage_df = pd.DataFrame(
        {
            "x": [xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin],
            "y": [ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax],
            "z": [zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax],
        }
    )

    xvtk, yvtk, zvtk, vtk_units = vtk_coordinate_arrays(cage_df)

    points = list(zip(xvtk, yvtk, zvtk))

    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"mixed_model_cage coordinate_units={vtk_units}\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")

        for x, y, z in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")

        f.write(f"\nLINES {len(lines)} {len(lines) * 3}\n")

        for i, j in lines:
            f.write(f"2 {i} {j}\n")

    print(f"[OK] Saved model cage VTK: {out_file}")
    print(f"[INFO] VTK coordinate units: {vtk_units}")


# ============================================================
# CATEGORICAL PLOTS
# ============================================================

def get_topview_nodes(mixed_df):
    """
    Build categorical 2D top-view model.

    Priority:
        high-rise no-fly > existing no-fly > flyable
    """
    df = mixed_df.copy()

    top = (
        df.groupby(["x", "y"], as_index=False)
        .agg(
            max_slowness=("slowness", "max"),
            existing_nofly=("existing_nofly", "max"),
            highrise_nofly=("highrise_nofly", "max"),
        )
    )

    top["slowness_class"] = 0
    top.loc[top["existing_nofly"] == 1, "slowness_class"] = 1
    top.loc[top["highrise_nofly"] == 1, "slowness_class"] = 2

    return top


def plot_2d_model_categorical(mixed_df, highrise_gdf, region, out_png):
    print("")
    print("========== PLOT 2D CATEGORICAL MODEL ==========")

    top = get_topview_nodes(mixed_df)

    fig = start_map(region, "Final mixed model: categorical 2D nodes")

    for cat in [0, 1, 2]:
        sub = top[top["slowness_class"] == cat].copy()

        if sub.empty:
            continue

        style = CATEGORY_STYLES[cat]

        fig.plot(
            x=sub["x"],
            y=sub["y"],
            style=style["style_2d"],
            fill=style["fill"],
            pen=None if cat == 0 else "0.1p,black",
            transparency=style["transparency_2d"],
            label=style["name"],
        )

        print(f"2D category {cat} {style['name']}: {len(sub):,} xy nodes")

    if PLOT_HIGHRISE_FOOTPRINT_ON_2D:
        plot_polygons_constant(
            fig,
            highrise_gdf,
            fill=HIGHRISE_FILL,
            pen=HIGHRISE_PEN,
            label="High-rise footprint",
        )

    plot_aoi_boundary(fig)

    fig.legend(
        position="JBL+jBL+o0.2c/0.2c",
        box="+gwhite@10+p0.5p,black",
    )

    fig.savefig(str(out_png), dpi=DPI)
    print(f"[OK] Saved 2D categorical model figure: {out_png}")


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


def plot_3d_model_categorical_0_100m(mixed_df, highrise_gdf, region, out_png):
    print("")
    print("========== PLOT 3D CATEGORICAL MODEL 0-100 M ==========")

    df = mixed_df.copy()

    df = df[
        (df["z"] >= PLOT_3D_Z_MIN_M)
        & (df["z"] <= PLOT_3D_Z_MAX_M)
    ].copy()

    if df.empty:
        raise ValueError(
            f"No model nodes found between z={PLOT_3D_Z_MIN_M} and z={PLOT_3D_Z_MAX_M} m."
        )

    # Plot z in km.
    df["z_km"] = df["z"] / 1000.0

    xmin, xmax, ymin, ymax = region
    zmin_km = PLOT_3D_Z_MIN_M / 1000.0
    zmax_km = PLOT_3D_Z_MAX_M / 1000.0

    region3d = [xmin, xmax, ymin, ymax, zmin_km, zmax_km]

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
            'WSneZ+t"Final mixed model: categorical 3D nodes 0-100 m"',
            "xaf+lLongitude",
            "yaf+lLatitude",
            "zaf+lElevation (km)",
        ],
    )

    category_max_points = {
        0: MAX_3D_FLYABLE_POINTS,
        1: MAX_3D_EXISTING_NOFLY_POINTS,
        2: MAX_3D_HIGHRISE_NOFLY_POINTS,
    }

    random_seeds = {
        0: 12345,
        1: 12346,
        2: 12347,
    }

    for cat in [0, 1, 2]:
        sub = df[df["slowness_class"] == cat].copy()
        sub = sample_for_3d_plot(
            sub,
            max_points=category_max_points[cat],
            random_state=random_seeds[cat],
        )

        if sub is None or sub.empty:
            continue

        style = CATEGORY_STYLES[cat]

        fig.plot3d(
            x=sub["x"],
            y=sub["y"],
            z=sub["z_km"],
            style=style["style_3d"],
            fill=style["fill"],
            pen=None if cat == 0 else "0.1p,black",
            transparency=style["transparency_3d"],
            perspective=PERSPECTIVE_3D,
            label=style["name"],
        )

        print(f"3D category {cat} {style['name']}: {len(sub):,} plotted nodes")

    if PLOT_HIGHRISE_FOOTPRINT_ON_3D:
        plot_polygon_3d_at_zkm(
            fig,
            highrise_gdf,
            z_km=zmin_km,
            pen=HIGHRISE_PEN,
            fill=HIGHRISE_FILL,
            label="High-rise footprint",
        )

        plot_polygon_3d_at_zkm(
            fig,
            highrise_gdf,
            z_km=zmax_km,
            pen="0.35p,green",
            fill=None,
            label=None,
        )

    poly_df = polygon_to_dataframe()
    fig.plot3d(
        x=poly_df["x"],
        y=poly_df["y"],
        z=[zmin_km] * len(poly_df),
        pen=POLYGON_PEN,
        perspective=PERSPECTIVE_3D,
        label="Hoa Lac boundary",
    )

    fig.legend(
        position="JBL+jBL+o0.2c/0.2c",
        box="+gwhite@10+p0.5p,black",
    )

    fig.savefig(str(out_png), dpi=DPI)
    print(f"[OK] Saved 3D categorical model figure: {out_png}")


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

    plot_2d_model_categorical(
        mixed_df=mixed_df,
        highrise_gdf=highrise,
        region=region,
        out_png=OUT_2D_MODEL_FIG,
    )

    plot_3d_model_categorical_0_100m(
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
    print(f"2D model figure:        {OUT_2D_MODEL_FIG}")
    print(f"3D model figure:        {OUT_3D_MODEL_FIG}")
    print(f"High-rise method:       {highrise_method}")
    print(f"No-fly slowness:        {NO_FLY_SLOWNESS}")
    print("VTK units:              x/y lonlat if geographic; z converted from m to km")


if __name__ == "__main__":
    main()
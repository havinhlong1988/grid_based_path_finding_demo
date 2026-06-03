#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot downloaded Hoa Lac XYZ/GPKG data using PyGMT.

Input folder:
    output_hoalac_opentopography/

Output folder:
    figures/01_download_xyz_opentopography/

Outputs:
    00_overview_map.png
        Road map + building polygons + Hoa Lac boundary

    01_<xyz_filename>.png
    02_<xyz_filename>.png
    ...
        Individual XYZ plots, first-come first-serve by file modification time.

Important:
    DEM / slope / TRI / buildings_grid are plotted using
    pygmt.surface + grdimage to avoid white dot/gap artifacts.

XYZ format:
    lon lat value
"""

# import xarray as xr
from matplotlib.path import Path as MplPath
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt


# ============================================================
# 0.0.0 USER INPUT PARAMETERS
# ============================================================

DATA_DIR = Path("output_hoalac_opentopography")
XYZ_DIR = DATA_DIR
FIG_DIR = Path("figures/01_download_xyz_opentopography")

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
POLYGON_LINE_STYLE = None       # "-" solid, "." dotted, "--" dashed
POLYGON_PEN = f"{POLYGON_LINE_WIDTH},{POLYGON_LINE_COLOR},{POLYGON_LINE_STYLE}"

# ----------------------------
# Road style for overview
# ----------------------------
ROAD_LINE_WIDTH = "0.5p"
ROAD_LINE_COLOR = "gray40"
ROAD_LINE_STYLE = None
ROAD_PEN = f"{ROAD_LINE_WIDTH},{ROAD_LINE_COLOR},{ROAD_LINE_STYLE}"

# Road fallback style
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
BUILDING_PEN_STYLE = "-"
BUILDING_PEN_TRANSPARENCY = 20
BUILDING_PEN = (
    f"{BUILDING_PEN_WIDTH},"
    f"{BUILDING_PEN_COLOR}@{BUILDING_PEN_TRANSPARENCY},"
    f"{BUILDING_PEN_STYLE}"
)

# ----------------------------
# Legend style
# ----------------------------
LEGEND_BOX_FILL = "white@10"
LEGEND_BOX_PEN = "0.5p,black"
LEGEND_POSITION = "JBL+jBL+o0.2c/0.2c"


# ============================================================
# 0.0.2 SURFACE / GRID PLOT PARAMETERS
# ============================================================

# Use surface/grdimage instead of point plotting for raster-like XYZ files
USE_SURFACE_FOR_RASTER_XYZ = True

# Temporary grid folder
TMP_GRID_DIR = FIG_DIR / "tmp_surface_grids"

# Surface spacing.
# None = automatically estimate from XYZ spacing
SURFACE_SPACING = None

# PyGMT surface tension.
# 0.25 is smooth but not too aggressive.
SURFACE_TENSION = 0.001

# Raster-like XYZ files that should be plotted by surface/grdimage
SURFACE_XYZ_KEYWORDS = [
    "terrain_dem",
    "terrain_slope",
    "terrain_ruggedness",
    "tri",
    "buildings_grid",
]

# Hide outside Hoa Lac polygon for surface maps
MASK_OUTSIDE_POLYGON = True

# ============================================================
# 0.0.3 ROAD CLASS STYLE PARAMETERS
# ============================================================

ROAD_CLASS_STYLE = {
    "motorway":      {"pen": "1.5p,red",        "label": "Motorway"},
    "trunk":         {"pen": "1.4p,orange",     "label": "Trunk"},
    "primary":       {"pen": "1.3p,yellow",     "label": "Primary"},
    "secondary":     {"pen": "1.1p,green",      "label": "Secondary"},
    "tertiary":      {"pen": "1.0p,cyan",       "label": "Tertiary"},
    "unclassified":  {"pen": "0.8p,gray40",     "label": "Unclassified"},
    "residential":   {"pen": "0.7p,blue",       "label": "Residential"},
    "service":       {"pen": "0.6p,gray60",     "label": "Service"},
    "living_street": {"pen": "0.6p,purple",     "label": "Living street"},
    "track":         {"pen": "0.6p,brown",      "label": "Track"},
    "path":          {"pen": "0.5p,magenta",    "label": "Path"},
    "footway":       {"pen": "0.5p,magenta",    "label": "Footway"},
    "cycleway":      {"pen": "0.5p,darkgreen",  "label": "Cycleway"},
    "pedestrian":    {"pen": "0.6p,darkgray",   "label": "Pedestrian"},
    "other":         {"pen": "0.5p,black",      "label": "Other"},
}

ROAD_LEGEND_POSITION = "JBR+jBR+o0.2c/0.2c"
ROAD_LEGEND_BOX = "+gwhite@10+p0.5p,black"


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


def make_map_title(text):
    """
    Format title text for PyGMT basemap frame.
    """
    return f'WSen+t"{text}"'


def read_xyz(xyz_file, keep_nan=False):
    """
    Read XYZ file.

    Expected format:
        lon lat value

    keep_nan=True is useful for road XYZ files if NaN rows exist.
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
# SURFACE HELPERS
# ============================================================

def is_surface_xyz_file(xyz_file):
    """
    Return True if this XYZ file should be plotted as a continuous surface.
    """
    name = xyz_file.stem.lower()
    return any(key in name for key in SURFACE_XYZ_KEYWORDS)


def estimate_xyz_spacing_for_surface(df):
    """
    Estimate lon/lat spacing from XYZ points.

    Returns PyGMT/GMT spacing string:
        "dx/dy"
    """
    xs = np.sort(df["lon"].unique())
    ys = np.sort(df["lat"].unique())

    dxs = np.diff(xs)
    dys = np.diff(ys)

    dxs = dxs[dxs > 0]
    dys = dys[dys > 0]

    if len(dxs) == 0 or len(dys) == 0:
        # Fallback spacing for Hoa Lac scale
        return "0.00025/0.00025"

    dx = float(np.median(dxs))
    dy = float(np.median(dys))

    return f"{dx}/{dy}"

def polygon_to_dataframe(polygon):
    """
    Convert polygon list [(lon, lat), ...] to DataFrame for pygmt.grdmask.
    """
    return pd.DataFrame(
        polygon,
        columns=["lon", "lat"],
    )


# def make_polygon_mask_grid(region, spacing, out_mask_grid):
#     """
#     Create mask grid from HOALAC_POLYGON.

#     Inside polygon  = 1
#     Outside polygon = NaN

#     This allows grdimage to hide outside polygon area.
#     """
#     out_mask_grid = Path(out_mask_grid)
#     out_mask_grid.parent.mkdir(parents=True, exist_ok=True)

#     poly_df = polygon_to_dataframe(HOALAC_POLYGON)

#     pygmt.grdmask(
#         data=poly_df,
#         region=region,
#         spacing=spacing,
#         inside=1,
#         outside=np.nan,
#         outgrid=str(out_mask_grid),
#     )

#     return out_mask_grid


# def apply_polygon_mask_to_grid(grid_file, mask_file, out_masked_grid):
#     """
#     Apply polygon mask to a surface grid.

#     masked_grid = grid * mask

#     Outside polygon becomes NaN.
#     """
#     grid_file = Path(grid_file)
#     mask_file = Path(mask_file)
#     out_masked_grid = Path(out_masked_grid)
#     out_masked_grid.parent.mkdir(parents=True, exist_ok=True)

#     pygmt.grdmath(
#         str(grid_file),
#         str(mask_file),
#         "MUL",
#         outgrid=str(out_masked_grid),
#     )

#     return out_masked_grid
def parse_spacing(spacing):
    """
    Parse GMT spacing string like:
        '0.00027778/0.00027778'
    """
    if isinstance(spacing, (int, float)):
        return float(spacing), float(spacing)

    parts = str(spacing).split("/")

    if len(parts) == 1:
        dx = float(parts[0])
        dy = dx
    else:
        dx = float(parts[0])
        dy = float(parts[1])

    return dx, dy


def adjust_region_to_spacing(region, spacing):
    """
    Adjust region so that:
        xmax - xmin = nx * dx
        ymax - ymin = ny * dy

    This avoids GMT surface error:
        Please select compatible -R and -I values
    grid_xyz = pygmt.grd2xyz(
        grid=str(grid_file),
        output_type="pandas",
    )

    """
    west, east, south, north = region
    dx, dy = parse_spacing(spacing)

    nx = int(np.ceil((east - west) / dx))
    ny = int(np.ceil((north - south) / dy))

    east_adj = west + nx * dx
    north_adj = south + ny * dy

    return [west, east_adj, south, north_adj]


def mask_grid_outside_polygon_python(grid_file, out_masked_grid, region, spacing):
    """
    Mask outside HOALAC_POLYGON without xarray/netcdf4.

    Workflow:
        1. Convert grid to XYZ using pygmt.grd2xyz
        2. Use matplotlib.path to find points inside polygon
        3. Set outside polygon values to NaN
        4. Convert masked XYZ back to grid using pygmt.xyz2grd

    This avoids:
        xarray.open_dataarray(...)
        netcdf4 / h5netcdf dependency problem
    """

    grid_file = Path(grid_file)
    out_masked_grid = Path(out_masked_grid)
    out_masked_grid.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Mask outside polygon for grid: {grid_file}")

    # Convert grid to XYZ dataframe
    grid_xyz = pygmt.grd2xyz(
        grid=str(grid_file),
        output_type="pandas",
    )

    # PyGMT may return columns as x/y/z or unnamed columns
    grid_xyz = grid_xyz.iloc[:, :3].copy()
    grid_xyz.columns = ["lon", "lat", "value"]

    polygon_path = MplPath(HOALAC_POLYGON)

    points = grid_xyz[["lon", "lat"]].to_numpy(dtype=float)
    inside = polygon_path.contains_points(points)

    # Outside polygon becomes NaN
    grid_xyz.loc[~inside, "value"] = np.nan

    # Rebuild grid
    pygmt.xyz2grd(
        data=grid_xyz[["lon", "lat", "value"]],
        region=region,
        spacing=spacing,
        outgrid=str(out_masked_grid),
    )

    print(f"[OK] Polygon mask applied: {out_masked_grid}")

    return out_masked_grid

def make_surface_grid_from_xyz(xyz_file, df, region, out_grid):
    """
    Create continuous grid from XYZ using pygmt.surface.

    Fixes:
        1. Adjust region to match spacing.
        2. Mask outside Hoa Lac polygon using Python/xarray.
        3. Avoid pygmt.grdmask(), which may not exist.
    """
    out_grid = Path(out_grid)
    out_grid.parent.mkdir(parents=True, exist_ok=True)

    if SURFACE_SPACING is None:
        spacing = estimate_xyz_spacing_for_surface(df)
    else:
        spacing = SURFACE_SPACING

    # Important fix for GMT surface:
    # region and spacing must be compatible.
    surface_region = adjust_region_to_spacing(region, spacing)

    print(f"[INFO] Surface grid for: {xyz_file.name}")
    print(f"[INFO] Surface spacing: {spacing}")
    print(f"[INFO] Original region: {region}")
    print(f"[INFO] Adjusted surface region: {surface_region}")

    sub = df.dropna(subset=["lon", "lat", "value"]).copy()

    sub = sub[
        (sub["lon"] >= surface_region[0]) &
        (sub["lon"] <= surface_region[1]) &
        (sub["lat"] >= surface_region[2]) &
        (sub["lat"] <= surface_region[3])
    ].copy()

    if sub.empty:
        raise ValueError(f"No valid XYZ points inside region for {xyz_file}")

    raw_grid = out_grid.with_name(out_grid.stem + "_raw.nc")

    # For building grid, preserve block-like values better
    if "buildings_grid" in xyz_file.stem.lower():
        pygmt.nearneighbor(
            data=sub[["lon", "lat", "value"]],
            region=surface_region,
            spacing=spacing,
            search_radius="0.001",
            outgrid=str(raw_grid),
        )
    else:
        pygmt.surface(
            data=sub[["lon", "lat", "value"]],
            region=surface_region,
            spacing=spacing,
            outgrid=str(raw_grid),
            tension=SURFACE_TENSION,
        )

    if MASK_OUTSIDE_POLYGON:
        masked_grid = mask_grid_outside_polygon_python(
            grid_file=raw_grid,
            out_masked_grid=out_grid,
            region=surface_region,
            spacing=spacing,
        )
        return masked_grid

    return raw_grid

# ============================================================
# PLOT INDIVIDUAL XYZ FILES
# ============================================================

def plot_xyz_surface(xyz_file, df, region, out_png):
    """
    Plot raster-like XYZ as continuous surface using pygmt.surface + grdimage.

    Good for:
        terrain_dem
        terrain_slope
        terrain_ruggedness_TRI
        buildings_grid
    """
    cmap, label = guess_plot_info(xyz_file.name)

    zmin, zmax = robust_zrange(df["value"].to_numpy())

    if "building" in xyz_file.name.lower():
        zmin = 0.0
        zmax = max(float(df["value"].max()), 1.0)

    cpt_step = (zmax - zmin) / 100.0

    if cpt_step <= 0 or not np.isfinite(cpt_step):
        cpt_step = 1.0

    TMP_GRID_DIR.mkdir(parents=True, exist_ok=True)

    grid_file = TMP_GRID_DIR / f"{xyz_file.stem}_surface.nc"

    grid_file = make_surface_grid_from_xyz(
        xyz_file=xyz_file,
        df=df,
        region=region,
        out_grid=grid_file,
    )

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
            make_map_title(xyz_file.stem),
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
        frame=f'af+l"{label}"',
        position="JBC+w10c/0.35c+h+o0c/0.8c",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"[OK] Saved surface figure: {out_png}")


def plot_xyz_points(xyz_file, df, region, out_png):
    """
    Plot XYZ points using PyGMT.

    Used for non-raster XYZ files:
        buildings_centroid
        buildings_vertices
    """
    cmap, label = guess_plot_info(xyz_file.name)

    zmin, zmax = robust_zrange(df["value"].to_numpy())

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
            make_map_title(xyz_file.stem),
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

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"[OK] Saved point figure: {out_png}")


# ============================================================
# ROAD CLASS PLOTTING
# ============================================================

def normalize_road_class(value):
    """
    Normalize OSM highway class.

    OSM highway can be string/list/tuple/None.
    """
    if value is None:
        return "other"

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "other"
        value = value[0]

    value = str(value)

    known = [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "service",
        "living_street",
        "track",
        "path",
        "footway",
        "cycleway",
        "pedestrian",
    ]

    if value in known:
        return value

    return "other"


def plot_lines_by_road_class(fig, roads_gdf):
    """
    Plot roads with different colors by OSM highway class.
    """
    if roads_gdf is None or roads_gdf.empty:
        return []

    roads = roads_gdf.to_crs("EPSG:4326").copy()

    if "highway_simple" in roads.columns:
        roads["road_class_plot"] = roads["highway_simple"].apply(normalize_road_class)
    elif "highway" in roads.columns:
        roads["road_class_plot"] = roads["highway"].apply(normalize_road_class)
    elif "road_code" in roads.columns:
        code_to_class = {
            1: "motorway",
            2: "trunk",
            3: "primary",
            4: "secondary",
            5: "tertiary",
            6: "unclassified",
            7: "residential",
            8: "service",
            9: "living_street",
            10: "track",
            11: "path",
            12: "footway",
            13: "cycleway",
            14: "pedestrian",
            99: "other",
        }
        roads["road_class_plot"] = roads["road_code"].map(code_to_class).fillna("other")
    else:
        roads["road_class_plot"] = "other"

    plot_order = [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "service",
        "living_street",
        "track",
        "path",
        "footway",
        "cycleway",
        "pedestrian",
        "other",
    ]

    used_classes = []

    for road_class in plot_order:
        sub = roads[roads["road_class_plot"] == road_class]

        if sub.empty:
            continue

        style = ROAD_CLASS_STYLE.get(
            road_class,
            ROAD_CLASS_STYLE["other"],
        )

        pen = style["pen"]

        for geom in sub.geometry:
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

                if coords.shape[0] < 2:
                    continue

                fig.plot(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    pen=pen,
                )

        used_classes.append(road_class)

    return used_classes


def add_road_class_legend(fig, used_classes):
    """
    Add legend for road classes that are actually present.
    """
    if not used_classes:
        return

    dummy_x = [-1000, -999]
    dummy_y = [-1000, -1000]

    for road_class in used_classes:
        style = ROAD_CLASS_STYLE.get(
            road_class,
            ROAD_CLASS_STYLE["other"],
        )

        fig.plot(
            x=dummy_x,
            y=dummy_y,
            pen=style["pen"],
            label=style["label"],
        )

    fig.legend(
        position=ROAD_LEGEND_POSITION,
        box=ROAD_LEGEND_BOX,
    )


def plot_roads_xyz(xyz_file, df, region, out_png):
    """
    Plot road map with different colors by OSM highway class.

    This uses roads_hoalac_clipped.gpkg instead of directly connecting
    road XYZ vertices, so each road segment remains separated correctly.
    """
    print("[INFO] Plotting roads from GPKG by road class")

    roads = None

    if ROADS_GPKG.exists():
        roads = gpd.read_file(ROADS_GPKG)
        print(f"[INFO] Roads loaded from GPKG: {len(roads)}")
        print(f"[INFO] Road columns: {list(roads.columns)}")

        if "highway_simple" in roads.columns:
            print("[INFO] Road classes from highway_simple:")
            print(roads["highway_simple"].astype(str).value_counts().head(20))
        elif "highway" in roads.columns:
            print("[INFO] Road classes from OSM highway:")
            print(roads["highway"].astype(str).value_counts().head(20))
    else:
        print(f"[WARN] Road GPKG not found: {ROADS_GPKG}")
        print("[WARN] Cannot plot class-colored roads without GPKG.")

    fig = pygmt.Figure()

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            make_map_title(f"{xyz_file.stem} by road class"),
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )

    used_classes = []

    if roads is not None and not roads.empty:
        used_classes = plot_lines_by_road_class(
            fig=fig,
            roads_gdf=roads,
        )
    else:
        print("[WARN] Fallback to one-color XYZ road plot.")

        # Fallback: use segment_id if the road XYZ has 4 columns.
        # Current read_xyz reads only 3 columns, so this fallback is minimal.
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

    add_road_class_legend(
        fig=fig,
        used_classes=used_classes,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"[OK] Saved road-class figure: {out_png}")


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
            make_map_title("Hoa Lac overview map: Buildings + Roads"),
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )

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

        out_png = FIG_DIR / f"{plot_id:02d}_{xyz_file.stem}.png"
        plot_id += 1

        if is_road_xyz:
            plot_roads_xyz(
                xyz_file=xyz_file,
                df=df,
                region=region,
                out_png=out_png,
            )

        elif USE_SURFACE_FOR_RASTER_XYZ and is_surface_xyz_file(xyz_file):
            plot_xyz_surface(
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
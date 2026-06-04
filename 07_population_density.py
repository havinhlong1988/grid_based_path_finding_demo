#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create building-density maps from OpenBuildingMap data.

This script includes BOTH density methods:

1. OBM polygon density
   - Uses real OpenBuildingMap building polygons:
       output/01_HoaLac_studies_area/openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg
   - Rasterizes building footprints
   - Gaussian smooths footprint raster
   - Good for true building-body density

2. Previous Gaussian point density
   - Uses OBM centroid + vertices XYZ files:
       input/02_data_senario1_no_velocity/buildings/obm*centroid*.xyz
       input/02_data_senario1_no_velocity/buildings/obm*vertices*.xyz
   - Combines points
   - Creates point-count grid
   - Gaussian smooths point-count grid
   - Good for checking dark-point-density logic

3. Combined density
   - Combines polygon density and Gaussian point density
   - Default mode: weighted_mean
   - Default weight:
       polygon = 0.40
       point   = 0.60

All density outside HOALAC_POLYGON is forced to 0.

Outputs:
    output/02_senario1_no_velocity/figures/obm_buildings_by_height.png
    output/02_senario1_no_velocity/figures/building_density_polygon_map.png
    output/02_senario1_no_velocity/figures/building_density_gaussian_points_map.png
    output/02_senario1_no_velocity/figures/building_density_combined_map.png

    output/02_senario1_no_velocity/building_density_polygon_grid.xyz
    output/02_senario1_no_velocity/building_density_gaussian_points_grid.xyz
    output/02_senario1_no_velocity/building_density_combined_grid.xyz
    output/02_senario1_no_velocity/building_density_summary.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt
import xarray as xr

from scipy.ndimage import gaussian_filter
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Polygon, mapping
from matplotlib.path import Path as MplPath


# ============================================================
# USER SETTINGS
# ============================================================

OBM_BUILDINGS_GPKG = Path(
    "output/01_HoaLac_studies_area/openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg"
)

OBM_XYZ_DIR = Path("input/02_data_senario1_no_velocity/buildings")

OUT_DIR = Path("output/02_senario1_no_velocity")
FIG_DIR = OUT_DIR / "figures"

OUT_OBM_HEIGHT_FIG = FIG_DIR / "obm_buildings_by_height.png"

OUT_POLYGON_DENSITY_FIG = FIG_DIR / "building_density_polygon_map.png"
OUT_GAUSSIAN_POINTS_DENSITY_FIG = FIG_DIR / "building_density_gaussian_points_map.png"
OUT_COMBINED_DENSITY_FIG = FIG_DIR / "building_density_combined_map.png"

OUT_POLYGON_DENSITY_XYZ = OUT_DIR / "building_density_polygon_grid.xyz"
OUT_GAUSSIAN_POINTS_DENSITY_XYZ = OUT_DIR / "building_density_gaussian_points_grid.xyz"
OUT_COMBINED_DENSITY_XYZ = OUT_DIR / "building_density_combined_grid.xyz"

OUT_SUMMARY = OUT_DIR / "building_density_summary.csv"

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


# ============================================================
# SWITCHES
# ============================================================

CREATE_POLYGON_DENSITY = True
CREATE_GAUSSIAN_POINT_DENSITY = True
CREATE_COMBINED_DENSITY = True

PLOT_OBM_HEIGHT_MAP = True
PLOT_POLYGON_DENSITY_MAP = True
PLOT_GAUSSIAN_POINT_DENSITY_MAP = True
PLOT_COMBINED_DENSITY_MAP = True

# If True, also plot height-colored OBM polygons on density map.
# Usually False because it hides density.
PLOT_HEIGHT_POLYGONS_ON_DENSITY = False


# ============================================================
# DENSITY SETTINGS
# ============================================================

DENSITY_GRID_SPACING_DEG = 0.00010

# Polygon-footprint density smoothing.
POLYGON_GAUSSIAN_SIGMA_CELLS = 2.0
POLYGON_DENSITY_CUTOFF = 0.02
RASTERIZE_ALL_TOUCHED = True

# Previous centroid+vertices Gaussian point density smoothing.
POINT_GAUSSIAN_SIGMA_CELLS = 2.0
POINT_DENSITY_CUTOFF = 0.02

# Combined density.
# Options:
#   "weighted_mean"
#   "max"
#   "sum"
COMBINED_DENSITY_MODE = "weighted_mean"

# More weight to previous Gaussian point density
# so combined map is visibly different from polygon-only map.
POLYGON_DENSITY_WEIGHT = 0.40
POINT_DENSITY_WEIGHT = 0.60


# ============================================================
# BUILDING HEIGHT / HIGH-RISE SETTINGS
# ============================================================

HEIGHT_COLUMN = "height_m"

PLOT_BUILDINGS_BY_HEIGHT = True

# If None, use percentile threshold.
HIGHRISE_HEIGHT_M = None
HIGHRISE_HEIGHT_PERCENTILE = 90

HIGHRISE_FILL = "green@15"
HIGHRISE_PEN = "0.45p,green"

BUILDING_FILL = "lightred@65"
BUILDING_PEN = "0.20p,black@35"
BUILDING_OUTLINE_PEN = "0.15p,black@45"
POLYGON_PEN = "1.4p,purple"

POINT_INPUT_STYLE = "c0.004c"

# -----------------------------------------------------------
CLEANUP_CPT_AND_TEMP_FILES = True
# -----------------------------------------------------------
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


def snap_region_to_spacing(region, spacing):
    xmin, xmax, ymin, ymax = region

    x0 = np.floor(xmin / spacing) * spacing
    x1 = np.ceil(xmax / spacing) * spacing
    y0 = np.floor(ymin / spacing) * spacing
    y1 = np.ceil(ymax / spacing) * spacing

    return [
        float(np.round(x0, 10)),
        float(np.round(x1, 10)),
        float(np.round(y0, 10)),
        float(np.round(y1, 10)),
    ]


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


def plot_aoi_boundary(fig, pen=POLYGON_PEN):
    poly_df = polygon_to_dataframe()

    fig.plot(
        x=poly_df["x"],
        y=poly_df["y"],
        pen=pen,
        fill=None,
        label="Hoa Lac boundary",
    )


def save_fig(fig, out_png):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=DPI)
    print(f"[OK] Saved: {out_png}")


def safe_polygons(geom):
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "Polygon":
        return [geom]

    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)

    return []


def make_polygon_mask_from_centers(lon_centers, lat_centers):
    poly_path = MplPath(HOALAC_POLYGON)

    lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)
    xy = np.column_stack([lon2d.ravel(), lat2d.ravel()])

    inside = poly_path.contains_points(xy, radius=1e-12)

    return inside.reshape(len(lat_centers), len(lon_centers))


# ============================================================
# LOAD OBM BUILDING POLYGONS
# ============================================================

def find_fallback_polygon_file():
    patterns = [
        "*building*.gpkg",
        "*buildings*.gpkg",
        "*obm*.gpkg",
        "*OBM*.gpkg",
        "*.gpkg",
        "*building*.geojson",
        "*buildings*.geojson",
        "*obm*.geojson",
        "*OBM*.geojson",
        "*.geojson",
        "*building*.shp",
        "*buildings*.shp",
        "*obm*.shp",
        "*OBM*.shp",
        "*.shp",
    ]

    for pattern in patterns:
        files = sorted(OBM_XYZ_DIR.glob(pattern))
        if files:
            return files[0]

    return None


def load_obm_buildings():
    if OBM_BUILDINGS_GPKG.exists():
        path = OBM_BUILDINGS_GPKG
    else:
        path = find_fallback_polygon_file()

    if path is None:
        raise FileNotFoundError(
            "No OBM polygon file found. Expected:\n"
            f"  {OBM_BUILDINGS_GPKG}\n"
            "or a GPKG/GeoJSON/SHP in:\n"
            f"  {OBM_XYZ_DIR}"
        )

    print("")
    print("========== LOAD OBM BUILDINGS ==========")
    print(f"OBM polygon file: {path}")

    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"OBM building polygon file is empty: {path}")

    if gdf.crs is None:
        print("[WARNING] OBM file has no CRS. Assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if gdf.empty:
        raise ValueError("No Polygon/MultiPolygon geometry found in OBM file.")

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
        print(f"Height column found:   {HEIGHT_COLUMN}")
        print(f"Height valid count:    {gdf[HEIGHT_COLUMN].notna().sum():,}")
        print(f"Height max:            {gdf[HEIGHT_COLUMN].max()}")
    else:
        print(f"[WARNING] Height column not found: {HEIGHT_COLUMN}")

    return gdf, path


# ============================================================
# LOAD PREVIOUS GAUSSIAN POINT INPUTS
# ============================================================

def read_xyz_auto(path: Path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )

    df = df.dropna(axis=1, how="all")
    ncol = df.shape[1]

    if ncol < 2:
        raise ValueError(f"File must have at least lon lat columns: {path}")

    if ncol == 2:
        df = df.iloc[:, :2]
        df.columns = ["x", "y"]
        df["z"] = 0.0
        df["value"] = 0.0

    elif ncol == 3:
        df = df.iloc[:, :3]
        df.columns = ["x", "y", "z"]
        df["value"] = 0.0

    elif ncol == 4:
        df = df.iloc[:, :4]
        df.columns = ["x", "y", "z", "value"]

    else:
        df = df.iloc[:, :ncol]
        names = ["x", "y", "z", "value"]
        names += [f"extra_{i}" for i in range(ncol - 4)]
        df.columns = names

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["x", "y"]).copy()

    if "z" not in df.columns:
        df["z"] = 0.0

    if "value" not in df.columns:
        df["value"] = 0.0

    return df


def infer_xy_are_lonlat(df):
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    return (
        np.nanmin(x) >= -180
        and np.nanmax(x) <= 180
        and np.nanmin(y) >= -90
        and np.nanmax(y) <= 90
    )


def unique_files(files):
    out = []
    seen = set()

    for f in files:
        if not f.is_file():
            continue

        rp = f.resolve()

        if rp in seen:
            continue

        out.append(f)
        seen.add(rp)

    return out


def collect_gaussian_point_files():
    preferred_patterns = [
        "obm*centroid*.xyz",
        "OBM*centroid*.xyz",
        "*obm*centroid*.xyz",
        "*OBM*centroid*.xyz",
        "obm*vertices*.xyz",
        "OBM*vertices*.xyz",
        "*obm*vertices*.xyz",
        "*OBM*vertices*.xyz",
        "obm*vertex*.xyz",
        "OBM*vertex*.xyz",
        "*obm*vertex*.xyz",
        "*OBM*vertex*.xyz",
    ]

    fallback_patterns = [
        "*centroid*.xyz",
        "*vertices*.xyz",
        "*vertex*.xyz",
    ]

    files = []

    for pattern in preferred_patterns:
        files.extend(sorted(OBM_XYZ_DIR.glob(pattern)))

    files = unique_files(files)

    if files:
        return files, "preferred obm centroid/vertices files"

    for pattern in fallback_patterns:
        files.extend(sorted(OBM_XYZ_DIR.glob(pattern)))

    files = unique_files(files)

    return files, "fallback centroid/vertices files"


def mask_points_inside_polygon(df):
    if df.empty:
        return df.copy()

    poly_path = MplPath(HOALAC_POLYGON)
    xy = df[["x", "y"]].to_numpy(dtype=float)

    inside = poly_path.contains_points(xy, radius=1e-12)

    return df.loc[inside].copy()


def load_combined_gaussian_points():
    files, mode = collect_gaussian_point_files()

    if not files:
        print("[WARNING] No centroid/vertices XYZ files found. Skip Gaussian point density.")
        return None, [], mode

    print("")
    print("========== LOAD GAUSSIAN POINT INPUTS ==========")
    print(f"Search mode: {mode}")
    print(f"Files used:  {len(files)}")

    parts = []

    for f in files:
        try:
            df = read_xyz_auto(f)

            if not infer_xy_are_lonlat(df):
                print(f"[WARNING] Skip non-lonlat file: {f}")
                continue

            df = df[["x", "y", "z", "value"]].copy()
            df["source_file"] = f.name

            name_lower = f.name.lower()

            if "centroid" in name_lower:
                df["source_type"] = "centroid"
            elif "vertices" in name_lower or "vertex" in name_lower:
                df["source_type"] = "vertices"
            else:
                df["source_type"] = "unknown"

            parts.append(df)

            print(f"  - {f.name}: {len(df):,} points")

        except Exception as exc:
            print(f"[WARNING] Failed to read {f}: {exc}")

    if not parts:
        print("[WARNING] No valid centroid/vertices points loaded. Skip Gaussian point density.")
        return None, files, mode

    all_df = pd.concat(parts, ignore_index=True)

    all_before = len(all_df)

    all_df = all_df.drop_duplicates(subset=["x", "y"]).copy()

    all_inside = mask_points_inside_polygon(all_df)

    print(f"Combined points before unique: {all_before:,}")
    print(f"Combined unique xy points:     {len(all_df):,}")
    print(f"Points inside polygon:         {len(all_inside):,}")
    print(f"Points outside polygon masked: {len(all_df) - len(all_inside):,}")

    if all_inside.empty:
        print("[WARNING] No centroid/vertices points inside polygon. Skip Gaussian point density.")
        return None, files, mode

    return all_inside, files, mode


# ============================================================
# PLOT BUILDING POLYGONS
# ============================================================

def plot_polygons_constant(fig, gdf, fill=None, pen=BUILDING_PEN, label=None):
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


def plot_buildings_by_height(fig, buildings, add_colorbar=True):
    if buildings is None or buildings.empty:
        return

    if not PLOT_BUILDINGS_BY_HEIGHT or HEIGHT_COLUMN not in buildings.columns:
        plot_polygons_constant(
            fig,
            buildings,
            fill=BUILDING_FILL,
            pen=BUILDING_PEN,
            label="Building polygon",
        )
        return

    valid = buildings.dropna(subset=[HEIGHT_COLUMN]).copy()

    if valid.empty:
        plot_polygons_constant(
            fig,
            buildings,
            fill=BUILDING_FILL,
            pen=BUILDING_PEN,
            label="Building polygon",
        )
        return

    zmin = float(valid[HEIGHT_COLUMN].quantile(0.02))
    zmax = float(valid[HEIGHT_COLUMN].quantile(0.98))

    if np.isclose(zmin, zmax):
        zmin = float(valid[HEIGHT_COLUMN].min())
        zmax = float(valid[HEIGHT_COLUMN].max())

    if np.isclose(zmin, zmax):
        zmin, zmax = 0.0, 10.0

    step = (zmax - zmin) / 100.0

    pygmt.makecpt(
        cmap="turbo",
        series=[zmin, zmax, step],
        continuous=True,
    )

    for _, row in valid.iterrows():
        geom = row.geometry
        value = row[HEIGHT_COLUMN]

        for poly in safe_polygons(geom):
            x, y = poly.exterior.xy

            fig.plot(
                x=list(x),
                y=list(y),
                fill=value,
                cmap=True,
                pen=BUILDING_PEN,
            )

    if add_colorbar:
        fig.colorbar(
            frame=f'af+l"Building height (m)"',
            position="JBC+w10c/0.35c+h+o0c/0.8c",
        )


def select_highrise_buildings(buildings):
    if buildings is None or buildings.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), "none"

    gdf = buildings.copy()

    if HEIGHT_COLUMN in gdf.columns:
        vals = pd.to_numeric(gdf[HEIGHT_COLUMN], errors="coerce")

        if vals.notna().sum() > 0 and vals.max() > 0:
            gdf["_height_for_plot"] = vals.fillna(0.0)

            if HIGHRISE_HEIGHT_M is not None:
                threshold = float(HIGHRISE_HEIGHT_M)
            else:
                positive = gdf.loc[gdf["_height_for_plot"] > 0, "_height_for_plot"]
                threshold = float(np.nanpercentile(positive, HIGHRISE_HEIGHT_PERCENTILE))

            highrise = gdf[gdf["_height_for_plot"] >= threshold].copy()
            method = f"{HEIGHT_COLUMN} >= {threshold:.2f} m"

            print("")
            print("========== HIGH-RISE SELECTION ==========")
            print(f"Method:          {method}")
            print(f"All buildings:   {len(gdf):,}")
            print(f"High-rise count: {len(highrise):,}")

            return highrise, method

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

    return highrise, method


# ============================================================
# DENSITY GRID CREATION
# ============================================================

def make_grid_geometry(region):
    xmin, xmax, ymin, ymax = region
    spacing = DENSITY_GRID_SPACING_DEG

    width = int(np.ceil((xmax - xmin) / spacing))
    height = int(np.ceil((ymax - ymin) / spacing))

    xmax2 = xmin + width * spacing
    ymax2 = ymin + height * spacing

    transform = from_origin(xmin, ymax2, spacing, spacing)

    lon_centers = xmin + (np.arange(width) + 0.5) * spacing
    lat_desc = ymax2 - (np.arange(height) + 0.5) * spacing
    lat_centers = lat_desc[::-1]

    exact_region = [xmin, xmax2, ymin, ymax2]

    return width, height, transform, lon_centers, lat_centers, exact_region


def create_polygon_density_grid(buildings, region):
    width, height, transform, lon_centers, lat_centers, exact_region = make_grid_geometry(region)

    print("")
    print("========== POLYGON DENSITY GRID ==========")
    print(f"Spacing degree: {DENSITY_GRID_SPACING_DEG}")
    print(f"Width x height: {width} x {height}")
    print(f"Region used:    {exact_region}")
    print(f"Gaussian sigma: {POLYGON_GAUSSIAN_SIGMA_CELLS}")
    print(f"Density cutoff: {POLYGON_DENSITY_CUTOFF}")

    shapes = []

    for geom in buildings.geometry:
        if geom is None or geom.is_empty:
            continue
        shapes.append((mapping(geom), 1.0))

    if not shapes:
        raise ValueError("No valid building polygon shapes to rasterize.")

    footprint = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        dtype="float32",
        all_touched=RASTERIZE_ALL_TOUCHED,
    )

    aoi_shape = [(mapping(get_hoalac_polygon()), 1)]

    aoi_mask = rasterize(
        aoi_shape,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)

    footprint[~aoi_mask] = 0.0

    if POLYGON_GAUSSIAN_SIGMA_CELLS > 0:
        density = gaussian_filter(
            footprint.astype(float),
            sigma=POLYGON_GAUSSIAN_SIGMA_CELLS,
            mode="constant",
            cval=0.0,
        )
    else:
        density = footprint.astype(float)

    density[~aoi_mask] = 0.0

    max_val = float(np.nanmax(density))

    if max_val > 0:
        density = density / max_val

    density[density < POLYGON_DENSITY_CUTOFF] = 0.0
    density[~aoi_mask] = 0.0

    density_asc = density[::-1, :]

    density_grid = xr.DataArray(
        density_asc,
        coords={
            "lat": lat_centers,
            "lon": lon_centers,
        },
        dims=("lat", "lon"),
        name="polygon_building_density",
    )

    print(f"Footprint nonzero cells: {int(np.sum(footprint > 0)):,}")
    print(f"Density nonzero cells:   {int(np.sum(density_asc > 0)):,}")
    print(f"Max density:             {float(np.nanmax(density_asc)):.4f}")
    print(f"Mean density:            {float(np.nanmean(density_asc)):.6f}")

    return density_grid, exact_region


def create_gaussian_point_density_grid(points_df, region):
    """
    Create Gaussian point-density grid from centroid + vertices points.

    Important:
        Use same width/height from make_grid_geometry().
        Use np.linspace for bin edges to avoid floating-point extra-bin error.
    """
    width, height, transform, lon_centers, lat_centers, exact_region = make_grid_geometry(region)

    xmin, xmax, ymin, ymax = exact_region

    print("")
    print("========== GAUSSIAN POINT DENSITY GRID ==========")
    print(f"Spacing degree:       {DENSITY_GRID_SPACING_DEG}")
    print(f"Width x height:       {width} x {height}")
    print(f"Region used:          {exact_region}")
    print(f"Gaussian sigma cells: {POINT_GAUSSIAN_SIGMA_CELLS}")
    print(f"Density cutoff:       {POINT_DENSITY_CUTOFF}")
    print(f"Input points:         {len(points_df):,}")

    lon_edges = np.linspace(xmin, xmax, width + 1)
    lat_edges = np.linspace(ymin, ymax, height + 1)

    counts_xy, _, _ = np.histogram2d(
        points_df["x"].to_numpy(),
        points_df["y"].to_numpy(),
        bins=[lon_edges, lat_edges],
    )

    counts = counts_xy.T

    if counts.shape != (height, width):
        raise ValueError(
            f"Unexpected point-density count shape: {counts.shape}, "
            f"expected {(height, width)}"
        )

    if POINT_GAUSSIAN_SIGMA_CELLS > 0:
        density = gaussian_filter(
            counts.astype(float),
            sigma=POINT_GAUSSIAN_SIGMA_CELLS,
            mode="constant",
            cval=0.0,
        )
    else:
        density = counts.astype(float)

    mask = make_polygon_mask_from_centers(lon_centers, lat_centers)

    if mask.shape != density.shape:
        raise ValueError(
            f"Mask shape mismatch: mask={mask.shape}, density={density.shape}"
        )

    density[~mask] = 0.0

    max_val = float(np.nanmax(density))

    if max_val > 0:
        density = density / max_val

    density[density < POINT_DENSITY_CUTOFF] = 0.0
    density[~mask] = 0.0

    density_grid = xr.DataArray(
        density,
        coords={
            "lat": lat_centers,
            "lon": lon_centers,
        },
        dims=("lat", "lon"),
        name="gaussian_point_building_density",
    )

    print(f"Max raw point count/cell: {float(np.nanmax(counts)):.3f}")
    print(f"Density nonzero cells:    {int(np.sum(density > 0)):,}")
    print(f"Max density:              {float(np.nanmax(density)):.4f}")
    print(f"Mean density:             {float(np.nanmean(density)):.6f}")

    return density_grid, exact_region


def combine_density_grids(polygon_grid, point_grid):
    if polygon_grid is None and point_grid is None:
        return None

    if polygon_grid is None:
        return point_grid.copy()

    if point_grid is None:
        return polygon_grid.copy()

    p = polygon_grid.to_numpy().astype(float)
    q = point_grid.to_numpy().astype(float)

    if p.shape != q.shape:
        raise ValueError(
            f"Density grid shapes differ: polygon={p.shape}, point={q.shape}. "
            "Use same region and spacing."
        )

    # Normalize both again so polygon density cannot dominate too much.
    pmax = float(np.nanmax(p))
    qmax = float(np.nanmax(q))

    if pmax > 0:
        p = p / pmax

    if qmax > 0:
        q = q / qmax

    if COMBINED_DENSITY_MODE == "max":
        combined = np.maximum(p, q)

    elif COMBINED_DENSITY_MODE == "weighted_mean":
        combined = (
            POLYGON_DENSITY_WEIGHT * p
            + POINT_DENSITY_WEIGHT * q
        )

        max_val = float(np.nanmax(combined))
        if max_val > 0:
            combined = combined / max_val

    elif COMBINED_DENSITY_MODE == "sum":
        combined = p + q

        max_val = float(np.nanmax(combined))
        if max_val > 0:
            combined = combined / max_val

    else:
        raise ValueError(f"Unknown COMBINED_DENSITY_MODE: {COMBINED_DENSITY_MODE}")

    # Keep outside polygon exactly zero.
    lon_centers = polygon_grid["lon"].to_numpy()
    lat_centers = polygon_grid["lat"].to_numpy()
    mask = make_polygon_mask_from_centers(lon_centers, lat_centers)

    combined[~mask] = 0.0

    combined_grid = xr.DataArray(
        combined,
        coords=polygon_grid.coords,
        dims=polygon_grid.dims,
        name="combined_building_density",
    )

    print("")
    print("========== COMBINED DENSITY GRID ==========")
    print(f"Mode:                  {COMBINED_DENSITY_MODE}")
    print(f"Polygon weight:        {POLYGON_DENSITY_WEIGHT}")
    print(f"Point weight:          {POINT_DENSITY_WEIGHT}")
    print(f"Nonzero density cells: {int(np.sum(combined > 0)):,}")
    print(f"Max density:           {float(np.nanmax(combined)):.4f}")
    print(f"Mean density:          {float(np.nanmean(combined)):.6f}")

    return combined_grid


def save_density_grid_xyz(density_grid, out_file):
    if density_grid is None:
        return

    lon = density_grid["lon"].to_numpy()
    lat = density_grid["lat"].to_numpy()
    den = density_grid.to_numpy()

    lon2d, lat2d = np.meshgrid(lon, lat)

    out_df = pd.DataFrame(
        {
            "lon": lon2d.ravel(),
            "lat": lat2d.ravel(),
            "density": den.ravel(),
        }
    )

    out_df.to_csv(
        out_file,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    print(f"[OK] Saved density grid xyz: {out_file}")


# ============================================================
# PLOTS
# ============================================================

def plot_obm_buildings_height_map(buildings, highrise, region, out_png):
    print("")
    print("========== PLOT OBM BUILDINGS HEIGHT MAP ==========")

    fig = start_map(region, "OpenBuildingMap buildings by height")

    plot_buildings_by_height(fig, buildings, add_colorbar=True)

    if highrise is not None and not highrise.empty:
        plot_polygons_constant(
            fig,
            highrise,
            fill=HIGHRISE_FILL,
            pen=HIGHRISE_PEN,
            label="High-rise building",
        )

    plot_aoi_boundary(fig)

    fig.plot(
        x=[-1000, -999],
        y=[-1000, -1000],
        pen=POLYGON_PEN,
        label="Hoa Lac boundary",
    )

    fig.legend(
        position="JBL+jBL+o0.2c/0.2c",
        box="+gwhite@10+p0.5p,black",
    )

    save_fig(fig, out_png)


def plot_density_map(
    density_grid,
    buildings,
    highrise,
    region,
    out_png,
    title,
    point_df=None,
    overlay_building_outlines=True,
    overlay_highrise=True,
    overlay_points=False,
):
    if density_grid is None:
        return

    print("")
    print(f"========== PLOT {title} ==========")

    cpt_file = OUT_DIR / f"{out_png.stem}.cpt"

    pygmt.makecpt(
        cmap="hot",
        series=[0, 1, 0.05],
        reverse=True,
        continuous=True,
        output=str(cpt_file),
    )

    fig = start_map(region, title)

    fig.grdimage(
        grid=density_grid,
        cmap=str(cpt_file),
        transparency=0,
    )

    # Optional height-colored building polygons.
    if PLOT_HEIGHT_POLYGONS_ON_DENSITY:
        plot_buildings_by_height(fig, buildings, add_colorbar=False)

    # Building outlines.
    if overlay_building_outlines:
        plot_polygons_constant(
            fig,
            buildings,
            fill=None,
            pen=BUILDING_OUTLINE_PEN,
            label="Building outline",
        )

    # High-rise green overlay.
    # Disabled for Gaussian point density map.
    if overlay_highrise and highrise is not None and not highrise.empty:
        plot_polygons_constant(
            fig,
            highrise,
            fill=HIGHRISE_FILL,
            pen=HIGHRISE_PEN,
            label="High-rise building",
        )

    # Centroid + vertices input points.
    if overlay_points and point_df is not None and not point_df.empty:
        fig.plot(
            x=point_df["x"],
            y=point_df["y"],
            style=POINT_INPUT_STYLE,
            fill="black",
            pen=None,
            transparency=60,
            label="Centroid + vertices",
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

    save_fig(fig, out_png)

def cleanup_cpt_and_temp_files():
    """
    Clean temporary CPT files and temporary PyGMT/GMT files after all figures are saved.
    Does NOT remove png, xyz, csv, gpkg, tif, shp, geojson.
    """
    if not CLEANUP_CPT_AND_TEMP_FILES:
        return

    print("")
    print("========== CLEANUP TEMP FILES ==========")

    cleanup_patterns = [
        OUT_DIR / "*.cpt",
        FIG_DIR / "*.cpt",
        FIG_DIR / "**/*.cpt",
        OUT_DIR / "gmt.history",
        FIG_DIR / "gmt.history",
        OUT_DIR / ".gmt*",
        FIG_DIR / ".gmt*",
    ]

    removed = 0

    for pattern in cleanup_patterns:
        for path in pattern.parent.glob(pattern.name):
            if path.is_file():
                try:
                    path.unlink()
                    removed += 1
                    print(f"[CLEAN] Removed: {path}")
                except Exception as exc:
                    print(f"[WARN] Could not remove {path}: {exc}")

    # Remove common temporary folders if they exist and are empty.
    temp_dirs = [
        FIG_DIR / "_tmp",
        FIG_DIR / "_tmp_reprojected_rasters",
        OUT_DIR / "_tmp",
    ]

    for d in temp_dirs:
        if d.exists() and d.is_dir():
            try:
                # Remove only if empty.
                d.rmdir()
                print(f"[CLEAN] Removed empty temp dir: {d}")
            except OSError:
                print(f"[INFO] Temp dir not empty, keep: {d}")

    print(f"[OK] Cleanup done. Removed files: {removed}")

# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_dirs()

    print("\n========== OBM BUILDING DENSITY ==========")

    region0 = get_region_from_polygon(padding=REGION_PADDING)
    region = snap_region_to_spacing(region0, DENSITY_GRID_SPACING_DEG)

    print(f"Plot region: {region}")

    buildings, building_file = load_obm_buildings()
    highrise, highrise_method = select_highrise_buildings(buildings)

    gaussian_points_df = None
    gaussian_point_files = []
    gaussian_point_mode = ""

    polygon_grid = None
    point_grid = None
    combined_grid = None

    if CREATE_GAUSSIAN_POINT_DENSITY:
        gaussian_points_df, gaussian_point_files, gaussian_point_mode = load_combined_gaussian_points()

    if CREATE_POLYGON_DENSITY:
        polygon_grid, polygon_region = create_polygon_density_grid(
            buildings=buildings,
            region=region,
        )

        save_density_grid_xyz(
            density_grid=polygon_grid,
            out_file=OUT_POLYGON_DENSITY_XYZ,
        )

    if CREATE_GAUSSIAN_POINT_DENSITY and gaussian_points_df is not None:
        point_grid, point_region = create_gaussian_point_density_grid(
            points_df=gaussian_points_df,
            region=region,
        )

        save_density_grid_xyz(
            density_grid=point_grid,
            out_file=OUT_GAUSSIAN_POINTS_DENSITY_XYZ,
        )

    if CREATE_COMBINED_DENSITY:
        combined_grid = combine_density_grids(
            polygon_grid=polygon_grid,
            point_grid=point_grid,
        )

        save_density_grid_xyz(
            density_grid=combined_grid,
            out_file=OUT_COMBINED_DENSITY_XYZ,
        )

    summary = pd.DataFrame(
        [
            {
                "obm_building_file": str(building_file),
                "buildings_used": len(buildings),
                "height_column": HEIGHT_COLUMN if HEIGHT_COLUMN in buildings.columns else "",
                "highrise_method": highrise_method,
                "highrise_count": len(highrise) if highrise is not None else 0,
                "gaussian_point_mode": gaussian_point_mode,
                "gaussian_point_files": "; ".join([f.name for f in gaussian_point_files]),
                "gaussian_points_inside_polygon": len(gaussian_points_df) if gaussian_points_df is not None else 0,
                "density_grid_spacing_degree": DENSITY_GRID_SPACING_DEG,
                "polygon_gaussian_sigma_cells": POLYGON_GAUSSIAN_SIGMA_CELLS,
                "polygon_density_cutoff": POLYGON_DENSITY_CUTOFF,
                "point_gaussian_sigma_cells": POINT_GAUSSIAN_SIGMA_CELLS,
                "point_density_cutoff": POINT_DENSITY_CUTOFF,
                "combined_density_mode": COMBINED_DENSITY_MODE,
                "polygon_density_weight": POLYGON_DENSITY_WEIGHT,
                "point_density_weight": POINT_DENSITY_WEIGHT,
                "rasterize_all_touched": RASTERIZE_ALL_TOUCHED,
                "outside_polygon_density": 0.0,
                "polygon_max_density": float(polygon_grid.max()) if polygon_grid is not None else np.nan,
                "point_max_density": float(point_grid.max()) if point_grid is not None else np.nan,
                "combined_max_density": float(combined_grid.max()) if combined_grid is not None else np.nan,
                "polygon_density_xyz": str(OUT_POLYGON_DENSITY_XYZ),
                "gaussian_points_density_xyz": str(OUT_GAUSSIAN_POINTS_DENSITY_XYZ),
                "combined_density_xyz": str(OUT_COMBINED_DENSITY_XYZ),
                "obm_height_figure": str(OUT_OBM_HEIGHT_FIG),
                "polygon_density_figure": str(OUT_POLYGON_DENSITY_FIG),
                "gaussian_points_density_figure": str(OUT_GAUSSIAN_POINTS_DENSITY_FIG),
                "combined_density_figure": str(OUT_COMBINED_DENSITY_FIG),
            }
        ]
    )

    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"[OK] Saved summary CSV: {OUT_SUMMARY}")

    if PLOT_OBM_HEIGHT_MAP:
        plot_obm_buildings_height_map(
            buildings=buildings,
            highrise=highrise,
            region=region,
            out_png=OUT_OBM_HEIGHT_FIG,
        )

    if PLOT_POLYGON_DENSITY_MAP and polygon_grid is not None:
        plot_density_map(
            density_grid=polygon_grid,
            buildings=buildings,
            highrise=highrise,
            region=region,
            out_png=OUT_POLYGON_DENSITY_FIG,
            title="OBM polygon footprint density",
            point_df=None,
            overlay_building_outlines=True,
            overlay_highrise=True,
            overlay_points=False,
        )

    if PLOT_GAUSSIAN_POINT_DENSITY_MAP and point_grid is not None:
        plot_density_map(
            density_grid=point_grid,
            buildings=buildings,
            highrise=highrise,
            region=region,
            out_png=OUT_GAUSSIAN_POINTS_DENSITY_FIG,
            title="Gaussian density from OBM centroids + vertices",
            point_df=gaussian_points_df,
            overlay_building_outlines=False,
            overlay_highrise=False,
            overlay_points=True,
        )

    if PLOT_COMBINED_DENSITY_MAP and combined_grid is not None:
        plot_density_map(
            density_grid=combined_grid,
            buildings=buildings,
            highrise=highrise,
            region=region,
            out_png=OUT_COMBINED_DENSITY_FIG,
            title="Combined density: polygon 40% + Gaussian point 60%",
            point_df=None,
            overlay_building_outlines=True,
            overlay_highrise=True,
            overlay_points=False,
        )
        
    cleanup_cpt_and_temp_files()

    print("\n========== DONE ==========")
    print(f"OBM height figure:             {OUT_OBM_HEIGHT_FIG}")
    print(f"Polygon density figure:        {OUT_POLYGON_DENSITY_FIG}")
    print(f"Gaussian point density figure: {OUT_GAUSSIAN_POINTS_DENSITY_FIG}")
    print(f"Combined density figure:       {OUT_COMBINED_DENSITY_FIG}")
    print(f"Polygon density xyz:           {OUT_POLYGON_DENSITY_XYZ}")
    print(f"Gaussian point density xyz:    {OUT_GAUSSIAN_POINTS_DENSITY_XYZ}")
    print(f"Combined density xyz:          {OUT_COMBINED_DENSITY_XYZ}")
    print(f"Summary CSV:                   {OUT_SUMMARY}")

    
    
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download Hoa Lac Hi-Tech Park building footprints from OpenBuildingMap (OBM)
via Google Earth Engine Community Catalog.

OBM GEE assets:
    Tile grid:
        projects/sat-io/open-datasets/OPEN-BUILDING-MAPS/open_buildings_grid

    Building tiles:
        projects/sat-io/open-datasets/OPEN-BUILDING-MAPS/tiles/building_<quadkey>

Outputs:
    output_hoalac_openbuildingmap/
    ├── hoalac_polygon.gpkg
    ├── obm_tile_grid_hoalac.gpkg
    ├── obm_buildings_raw.geojson
    ├── obm_buildings_hoalac_clipped.gpkg
    ├── obm_buildings_centroid_hoalac.xyz
    ├── obm_buildings_vertices_hoalac.xyz
    ├── obm_buildings_grid_hoalac.xyz
    └── obm_summary.csv

XYZ formats:
    centroid:
        lon lat height_m

    vertices:
        lon lat height_m

    grid:
        lon lat height_m
"""

from pathlib import Path
import json
import time
import warnings

import ee
import geemap
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin, xy
from shapely.geometry import Polygon

# ============================================================
# GOOGLE EARTH ENGINE SETUP CHECK
# ============================================================

ASK_GEE_SETUP_BEFORE_RUN = True

# Optional. If you already know your project ID, set it here.
# Example:
# GEE_PROJECT_ID = "utm-gee-hoalac"
GEE_PROJECT_ID = "starlit-road-461909-g4"

# ============================================================
# USER INPUT PARAMETERS
# ============================================================

OUTDIR = Path("output_hoalac_openbuildingmap")

# Hoa Lac polygon, lon/lat
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

# OBM GEE assets
OBM_GRID_ASSET = (
    "projects/sat-io/open-datasets/OPEN-BUILDING-MAPS/open_buildings_grid"
)

OBM_TILE_PREFIX = (
    "projects/sat-io/open-datasets/OPEN-BUILDING-MAPS/tiles/building_"
)

# Export settings
EXPORT_GEOJSON = OUTDIR / "obm_buildings_raw.geojson"
EXPORT_TIMEOUT_SEC = 600

# If the area has too many buildings, reduce first for testing.
# None = no limit.
MAX_BUILDINGS_TO_DOWNLOAD = None

# Keep only buildings that intersect Hoa Lac polygon after local clipping.
LOCAL_CLIP_TO_POLYGON = True

# Default height if OBM height is missing
DEFAULT_BUILDING_HEIGHT = 6.0

# Grid export resolution for building height raster/grid XYZ
# Unit: degree. 0.0001 deg is roughly 10-11 m near Vietnam.
BUILDING_GRID_RES_DEG = 0.0001

# If True, rasterize all touched cells of a footprint
BUILDING_GRID_ALL_TOUCHED = True


# ============================================================
# OUTPUT FILES
# ============================================================

POLYGON_GPKG = OUTDIR / "hoalac_polygon.gpkg"
TILE_GRID_GPKG = OUTDIR / "obm_tile_grid_hoalac.gpkg"
BUILDINGS_GPKG = OUTDIR / "obm_buildings_hoalac_clipped.gpkg"

CENTROID_XYZ = OUTDIR / "obm_buildings_centroid_hoalac.xyz"
VERTICES_XYZ = OUTDIR / "obm_buildings_vertices_hoalac.xyz"
GRID_XYZ = OUTDIR / "obm_buildings_grid_hoalac.xyz"
SUMMARY_CSV = OUTDIR / "obm_summary.csv"

# ============================================================
# API for OpenBuildingMap download and processing
# ============================================================

def ask_google_earth_engine_setup():
    """
    Ask user whether Google Earth Engine API has already been set up.

    If yes:
        continue script.

    If no:
        print setup guide and stop before using Earth Engine.
    """
    if not ASK_GEE_SETUP_BEFORE_RUN:
        return

    print("\n========== GOOGLE EARTH ENGINE SETUP CHECK ==========")
    print("This script needs Google Earth Engine API to download OpenBuildingMap data.")
    print("")

    answer = input(
        "Do you already set up Google Earth Engine API and project ID? [y/n]: "
    ).strip().lower()

    if answer in ["y", "yes"]:
        print("[OK] Continue with Google Earth Engine.\n")
        return

    if answer in ["n", "no"]:
        print_google_earth_engine_setup_guide()
        raise SystemExit(
            "\n[STOP] Please finish Google Earth Engine setup first, then run this script again.\n"
        )

    raise SystemExit(
        "\n[STOP] Invalid answer. Please type y or n, then press Enter.\n"
    )


def print_google_earth_engine_setup_guide():
    """
    Print guide for setting up Google Earth Engine API.
    """
    guide = """
============================================================
GOOGLE EARTH ENGINE API SETUP GUIDE
============================================================

1. Activate your conda environment:

   conda activate utm

2. Install Earth Engine Python tools if not installed:

   conda install -c conda-forge earthengine-api geemap -y

3. Authenticate your Google account:

   earthengine authenticate

   A browser will open. Log in with your Google account.

4. Get or create a Google Cloud Project ID.

   In Google Cloud Console:
   - Open project selector
   - Click "New Project" if needed
   - Copy the "Project ID", not project name, not project number

   Example Project ID:
   utm-gee-hoalac

5. Enable Google Earth Engine API for that project.

   In Google Cloud Console:
   - APIs & Services
   - Library
   - Search "Google Earth Engine API"
   - Click Enable

6. Register/configure Earth Engine access.

   If you do not want billing, choose:
   Community tier

   Do NOT choose Contributor tier unless you accept billing requirement.

7. Set your Earth Engine project in terminal:

   earthengine set_project YOUR_PROJECT_ID

   Example:

   earthengine set_project utm-gee-hoalac

8. Test Earth Engine:

   earthengine ls

   If no error appears, setup is OK.

9. Run this script again:

   python 01_download_data_openbuildingmaps_xyz.py

Optional:
You can also edit this script and set:

   GEE_PROJECT_ID = "YOUR_PROJECT_ID"

============================================================
"""
    print(guide)

# ============================================================
# BASIC GEOMETRY
# ============================================================

def make_hoalac_polygon_gdf():
    """
    Create Hoa Lac polygon GeoDataFrame.
    """
    poly = Polygon(HOALAC_POLYGON)

    if not poly.is_valid:
        poly = poly.buffer(0)

    gdf = gpd.GeoDataFrame(
        {"name": ["Hoa_Lac_HiTech_Park_approx"]},
        geometry=[poly],
        crs="EPSG:4326",
    )

    return gdf


def make_ee_polygon():
    """
    Create Earth Engine polygon geometry.
    """
    return ee.Geometry.Polygon([HOALAC_POLYGON], proj="EPSG:4326", geodesic=False)


# ============================================================
# EARTH ENGINE INITIALIZATION
# ============================================================

def initialize_earth_engine():
    """
    Initialize Earth Engine.

    Uses GEE_PROJECT_ID if provided.
    Otherwise uses project previously set by:
        earthengine set_project YOUR_PROJECT_ID
    """
    project_is_set = (
        GEE_PROJECT_ID is not None
        and str(GEE_PROJECT_ID).strip() != ""
        and GEE_PROJECT_ID != "YOUR_PROJECT_ID_HERE"
    )

    try:
        if project_is_set:
            ee.Initialize(project=GEE_PROJECT_ID)
            print(f"[OK] Earth Engine initialized with project: {GEE_PROJECT_ID}")
        else:
            ee.Initialize()
            print("[OK] Earth Engine initialized with default configured project.")

    except Exception as e:
        raise RuntimeError(
            "\n[ERROR] Earth Engine initialization failed.\n\n"
            f"Original error:\n{e}\n\n"
            "Most common fixes:\n"
            "1. Run: earthengine authenticate\n"
            "2. Run: earthengine set_project YOUR_PROJECT_ID\n"
            "3. Enable Google Earth Engine API for that project\n"
            "4. Register the project as Community tier if you do not want billing\n"
            "5. Or edit this script and set:\n"
            "   GEE_PROJECT_ID = 'YOUR_PROJECT_ID'\n"
        )


# ============================================================
# OBM TILE SELECTION
# ============================================================

def get_obm_tiles_for_aoi(aoi_ee):
    """
    Select OBM tile-grid features intersecting the AOI.
    """
    grid = ee.FeatureCollection(OBM_GRID_ASSET)

    tiles = grid.filterBounds(aoi_ee)

    n_tiles = tiles.size().getInfo()
    print(f"[INFO] Number of OBM tiles intersecting AOI: {n_tiles}")

    if n_tiles == 0:
        raise RuntimeError("No OBM tiles found for AOI.")

    quadkeys = tiles.aggregate_array("quadkey").getInfo()

    if quadkeys is None or len(quadkeys) == 0:
        raise RuntimeError("Could not read quadkey list from OBM tile grid.")

    quadkeys = [str(q) for q in quadkeys]

    print("[INFO] OBM quadkeys:")
    for q in quadkeys:
        print(f"       {q}")

    return tiles, quadkeys


def export_tile_grid_to_gpkg(tiles_ee, out_gpkg):
    """
    Export selected tile grid to local GeoPackage for checking.
    """
    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    tmp_geojson = out_gpkg.with_suffix(".geojson")

    print(f"[INFO] Exporting OBM tile grid to: {tmp_geojson}")

    geemap.ee_export_vector(
        tiles_ee,
        filename=str(tmp_geojson),
    )

    gdf = gpd.read_file(tmp_geojson).to_crs("EPSG:4326")
    gdf.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved OBM tile grid: {out_gpkg}")

    return gdf


# ============================================================
# OBM BUILDING DOWNLOAD
# ============================================================

def merge_obm_building_tiles(quadkeys, aoi_ee):
    """
    Load all OBM building tiles for selected quadkeys and merge them.
    """
    merged = ee.FeatureCollection([])

    for q in quadkeys:
        asset_id = f"{OBM_TILE_PREFIX}{q}"

        print(f"[INFO] Loading OBM tile: {asset_id}")

        tile_fc = ee.FeatureCollection(asset_id).filterBounds(aoi_ee)

        try:
            n_tile = tile_fc.size().getInfo()
            print(f"       buildings intersecting AOI: {n_tile}")
        except Exception as e:
            print(f"[WARN] Could not check tile size for {q}: {e}")
            n_tile = None

        merged = merged.merge(tile_fc)

    merged = merged.filterBounds(aoi_ee)

    if MAX_BUILDINGS_TO_DOWNLOAD is not None:
        merged = merged.limit(int(MAX_BUILDINGS_TO_DOWNLOAD))

    n_total = merged.size().getInfo()
    print(f"[INFO] Total OBM buildings before local clip: {n_total}")

    if n_total == 0:
        raise RuntimeError("No OBM buildings found for AOI.")

    return merged


def export_obm_buildings_to_geojson(buildings_ee, out_geojson):
    """
    Export OBM buildings to local GeoJSON using geemap.

    For a small AOI like Hoa Lac, this should usually work directly.
    """
    out_geojson = Path(out_geojson)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Exporting OBM buildings to local GeoJSON:")
    print(f"       {out_geojson}")

    geemap.ee_export_vector(
        buildings_ee,
        filename=str(out_geojson),
    )

    if not out_geojson.exists() or out_geojson.stat().st_size == 0:
        raise RuntimeError(f"Export failed or empty file: {out_geojson}")

    print(f"[OK] Saved raw OBM GeoJSON: {out_geojson}")


# ============================================================
# LOCAL CLEANING
# ============================================================

def parse_height_value(value):
    """
    Convert OBM height-like field to meters.

    The OBM page documents height attributes such as:
        height
        HHT:<meters>
        H:<stories>
        HBET:<range>
        HAPP:<approx stories>

    This function tries to be permissive.
    """
    if value is None or pd.isna(value):
        return np.nan

    raw = str(value).strip()

    if raw == "" or raw.lower() in ["none", "nan", "null"]:
        return np.nan

    # Simple numeric height
    try:
        return float(raw)
    except ValueError:
        pass

    # Height in meters, e.g. HHT:10.0
    if raw.startswith("HHT:"):
        try:
            return float(raw.split(":", 1)[1])
        except ValueError:
            return np.nan

    # Exact number of stories, e.g. H:1
    # Convert stories to approximate meters.
    if raw.startswith("H:"):
        try:
            stories = float(raw.split(":", 1)[1])
            return stories * 3.0
        except ValueError:
            return np.nan

    # Approx stories
    if raw.startswith("HAPP:"):
        try:
            stories = float(raw.split(":", 1)[1])
            return stories * 3.0
        except ValueError:
            return np.nan

    # Range of stories, e.g. HBET:1-3
    if raw.startswith("HBET:"):
        try:
            range_txt = raw.split(":", 1)[1]
            parts = range_txt.split("-")
            vals = [float(p) for p in parts if p.strip() != ""]
            if len(vals) > 0:
                return float(np.mean(vals)) * 3.0
        except ValueError:
            return np.nan

    return np.nan


def find_height_column(gdf):
    """
    Find likely OBM height column.
    """
    candidates = [
        "height",
        "height_m",
        "building_height",
        "stories",
        "taxonomy",
        "taxonomy_height",
        "gem_taxonomy",
    ]

    for c in candidates:
        if c in gdf.columns:
            return c

    # Fallback: any column containing height
    for c in gdf.columns:
        if "height" in c.lower():
            return c

    return None


def clean_and_clip_obm_buildings(raw_geojson, polygon_gdf, out_gpkg):
    """
    Read raw OBM GeoJSON, clean attributes, clip to Hoa Lac polygon.
    """
    raw_geojson = Path(raw_geojson)

    print(f"[INFO] Reading raw OBM buildings: {raw_geojson}")

    gdf = gpd.read_file(raw_geojson)

    if gdf.empty:
        raise RuntimeError("Raw OBM GeoJSON is empty.")

    gdf = gdf.to_crs("EPSG:4326").copy()

    # Keep only polygon geometries
    gdf = gdf[
        gdf.geometry.notna()
        & (~gdf.geometry.is_empty)
        & gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if gdf.empty:
        raise RuntimeError("No polygon building geometries found in OBM output.")

    print(f"[INFO] Raw polygon buildings: {len(gdf)}")
    print(f"[INFO] Raw OBM columns:")
    print(list(gdf.columns))

    height_col = find_height_column(gdf)

    if height_col is not None:
        print(f"[INFO] Height-like column detected: {height_col}")
        gdf["height_m"] = gdf[height_col].apply(parse_height_value)
    else:
        print("[WARN] No height-like column detected.")
        gdf["height_m"] = np.nan

    gdf["height_m"] = gdf["height_m"].fillna(DEFAULT_BUILDING_HEIGHT)

    # Clip locally with exact polygon
    if LOCAL_CLIP_TO_POLYGON:
        polygon = polygon_gdf.to_crs("EPSG:4326").copy()
        gdf = gpd.clip(gdf, polygon).reset_index(drop=True)

    if gdf.empty:
        raise RuntimeError("No OBM buildings remain after local clipping.")

    # Add area in m2
    projected_crs = gdf.estimate_utm_crs()
    gdf_m = gdf.to_crs(projected_crs)
    gdf["area_m2"] = gdf_m.geometry.area.to_numpy()

    # Ensure useful columns exist
    for col in ["source", "occupancy"]:
        if col not in gdf.columns:
            gdf[col] = "unknown"

    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved clipped OBM buildings: {out_gpkg}")
    print(f"[INFO] Clipped OBM buildings: {len(gdf)}")

    return gdf


# ============================================================
# XYZ EXPORTS
# ============================================================

def save_building_centroids_xyz(gdf, out_xyz):
    """
    Save one point per building:
        lon lat height_m
    """
    out_xyz = Path(out_xyz)

    if gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] Empty centroid XYZ saved: {out_xyz}")
        return

    gdf = gdf.copy().reset_index(drop=True)

    projected_crs = gdf.estimate_utm_crs()
    gdf_projected = gdf.to_crs(projected_crs).reset_index(drop=True)

    centroids = gdf_projected.geometry.centroid

    centroid_gdf = gpd.GeoDataFrame(
        {"height_m": gdf_projected["height_m"].to_numpy()},
        geometry=gpd.GeoSeries(centroids.to_numpy(), crs=projected_crs),
        crs=projected_crs,
    ).to_crs("EPSG:4326")

    df = pd.DataFrame({
        "lon": centroid_gdf.geometry.x.to_numpy(),
        "lat": centroid_gdf.geometry.y.to_numpy(),
        "height_m": centroid_gdf["height_m"].to_numpy(),
    })

    df.to_csv(
        out_xyz,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    print(f"[OK] Saved centroid XYZ: {out_xyz}")


def save_building_vertices_xyz(gdf, out_xyz):
    """
    Save building polygon vertices:
        lon lat height_m
    """
    out_xyz = Path(out_xyz)

    if gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] Empty vertices XYZ saved: {out_xyz}")
        return

    gdf = gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)

    records = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        height = float(row.get("height_m", DEFAULT_BUILDING_HEIGHT))

        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            continue

        for poly in polygons:
            for x, y in poly.exterior.coords:
                records.append((x, y, height))

    df = pd.DataFrame(records, columns=["lon", "lat", "height_m"])

    df.to_csv(
        out_xyz,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    print(f"[OK] Saved vertices XYZ: {out_xyz}")


def rasterize_buildings_to_grid_xyz(gdf, polygon_gdf, out_xyz):
    """
    Rasterize building height to regular lon/lat grid and save XYZ:
        lon lat height_m
    """
    out_xyz = Path(out_xyz)

    west, south, east, north = polygon_gdf.total_bounds

    res = BUILDING_GRID_RES_DEG

    width = int(np.ceil((east - west) / res))
    height = int(np.ceil((north - south) / res))

    transform = from_origin(
        west,
        north,
        res,
        res,
    )

    if gdf.empty:
        grid = np.zeros((height, width), dtype="float32")
    else:
        shapes = [
            (geom, float(h))
            for geom, h in zip(gdf.geometry, gdf["height_m"])
            if geom is not None and not geom.is_empty
        ]

        grid = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0.0,
            dtype="float32",
            all_touched=BUILDING_GRID_ALL_TOUCHED,
        )

    rows, cols = np.where(np.isfinite(grid))
    xs, ys = xy(transform, rows, cols, offset="center")
    vals = grid[rows, cols]

    df = pd.DataFrame({
        "lon": xs,
        "lat": ys,
        "height_m": vals,
    })

    df.to_csv(
        out_xyz,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    print(f"[OK] Saved building grid XYZ: {out_xyz}")


# ============================================================
# SUMMARY
# ============================================================

def save_summary(gdf, out_csv):
    """
    Save simple OBM summary table.
    """
    rows = []

    rows.append({
        "metric": "n_buildings",
        "value": len(gdf),
    })

    rows.append({
        "metric": "height_m_mean",
        "value": float(gdf["height_m"].mean()),
    })

    rows.append({
        "metric": "height_m_median",
        "value": float(gdf["height_m"].median()),
    })

    rows.append({
        "metric": "height_m_min",
        "value": float(gdf["height_m"].min()),
    })

    rows.append({
        "metric": "height_m_max",
        "value": float(gdf["height_m"].max()),
    })

    rows.append({
        "metric": "area_m2_sum",
        "value": float(gdf["area_m2"].sum()),
    })

    if "source" in gdf.columns:
        for key, val in gdf["source"].astype(str).value_counts().items():
            rows.append({
                "metric": f"source_{key}",
                "value": int(val),
            })

    if "occupancy" in gdf.columns:
        for key, val in gdf["occupancy"].astype(str).value_counts().items():
            rows.append({
                "metric": f"occupancy_{key}",
                "value": int(val),
            })

    df = pd.DataFrame(rows)

    df.to_csv(out_csv, index=False)

    print(f"[OK] Saved summary: {out_csv}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("\n========== HOA LAC OPENBUILDINGMAP DOWNLOAD ==========")

    ask_google_earth_engine_setup()

    # --------------------------------------------------------
    # 1. Local polygon
    # --------------------------------------------------------
    polygon_gdf = make_hoalac_polygon_gdf()
    polygon_gdf.to_file(POLYGON_GPKG, driver="GPKG")

    print(f"[OK] Saved Hoa Lac polygon: {POLYGON_GPKG}")

    # --------------------------------------------------------
    # 2. Earth Engine init
    # --------------------------------------------------------
    initialize_earth_engine()

    aoi_ee = make_ee_polygon()

    # --------------------------------------------------------
    # 3. Select OBM tiles
    # --------------------------------------------------------
    tiles_ee, quadkeys = get_obm_tiles_for_aoi(aoi_ee)

    export_tile_grid_to_gpkg(
        tiles_ee=tiles_ee,
        out_gpkg=TILE_GRID_GPKG,
    )

    # --------------------------------------------------------
    # 4. Merge OBM building tiles
    # --------------------------------------------------------
    buildings_ee = merge_obm_building_tiles(
        quadkeys=quadkeys,
        aoi_ee=aoi_ee,
    )

    # --------------------------------------------------------
    # 5. Export EE features to local GeoJSON
    # --------------------------------------------------------
    export_obm_buildings_to_geojson(
        buildings_ee=buildings_ee,
        out_geojson=EXPORT_GEOJSON,
    )

    # --------------------------------------------------------
    # 6. Local clean, height parse, exact clipping
    # --------------------------------------------------------
    buildings_gdf = clean_and_clip_obm_buildings(
        raw_geojson=EXPORT_GEOJSON,
        polygon_gdf=polygon_gdf,
        out_gpkg=BUILDINGS_GPKG,
    )

    # --------------------------------------------------------
    # 7. Export XYZ products
    # --------------------------------------------------------
    save_building_centroids_xyz(
        gdf=buildings_gdf,
        out_xyz=CENTROID_XYZ,
    )

    save_building_vertices_xyz(
        gdf=buildings_gdf,
        out_xyz=VERTICES_XYZ,
    )

    rasterize_buildings_to_grid_xyz(
        gdf=buildings_gdf,
        polygon_gdf=polygon_gdf,
        out_xyz=GRID_XYZ,
    )

    # --------------------------------------------------------
    # 8. Summary
    # --------------------------------------------------------
    save_summary(
        gdf=buildings_gdf,
        out_csv=SUMMARY_CSV,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------
    print("\n========== DONE ==========")
    print(f"Output folder: {OUTDIR.resolve()}")
    print("")
    print("Important files:")
    print(f"  OBM buildings GPKG:     {BUILDINGS_GPKG}")
    print(f"  OBM centroid XYZ:       {CENTROID_XYZ}")
    print(f"  OBM vertices XYZ:       {VERTICES_XYZ}")
    print(f"  OBM grid XYZ:           {GRID_XYZ}")
    print(f"  OBM summary CSV:        {SUMMARY_CSV}")
    print("")
    print("You can plot this with your existing PyGMT script by changing:")
    print(f"  DATA_DIR = Path('{OUTDIR}')")
    print(f"  BUILDINGS_GPKG = DATA_DIR / 'obm_buildings_hoalac_clipped.gpkg'")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download Hoa Lac Hi-Tech Park DEM/topography from OpenTopography,
then download buildings and roads from OpenStreetMap.

Outputs:
    output_hoalac_opentopography/
    ├── hoalac_polygon.gpkg
    ├── dem_opentopography_bbox.tif
    ├── dem_hoalac_clipped.tif
    ├── terrain_dem_hoalac.xyz
    ├── terrain_slope_hoalac.xyz
    ├── terrain_ruggedness_TRI_hoalac.xyz
    ├── buildings_bbox.gpkg
    ├── buildings_hoalac_clipped.gpkg
    ├── buildings_centroid_hoalac.xyz
    ├── buildings_vertices_hoalac.xyz
    ├── buildings_grid_hoalac.xyz
    ├── roads_bbox.gpkg
    ├── roads_hoalac_clipped.gpkg
    └── roads_vertices_hoalac.xyz

XYZ format:
    lon lat value

DEM source:
    OpenTopography Global DEM API
    https://portal.opentopography.org/API/globaldem

Building/road source:
    OpenStreetMap through OSMnx
"""

from pathlib import Path
import warnings
import requests

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.transform import xy
from shapely.geometry import Polygon
from scipy.ndimage import generic_filter


# ============================================================
# USER INPUT PARAMETERS
# ============================================================

# Hoa Lac polygon, format: lon, lat
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

# Output folder
OUTDIR = "output_hoalac_opentopography"

# Padding around polygon bbox for download
BBOX_PADDING_DEG = 0.002

# ============================================================
# OPENTOPOGRAPHY PARAMETERS
# ============================================================

# OpenTopography DEM type.
# Common options:
#   SRTMGL1  = SRTM 30 m
#   SRTMGL3  = SRTM 90 m
#   AW3D30   = ALOS World 3D 30 m
#   COP30    = Copernicus DEM 30 m
#   COP90    = Copernicus DEM 90 m
#   NASADEM  = NASADEM
DEMTYPE = "COP30"

# OpenTopography API key is required for many downloads.
# Get it from https://opentopography.org/
OPENTOPOGRAPHY_API_KEY = "9b13849a6bd3486c4ed72960d230a366"

# If True, stop when API key is still not set.
REQUIRE_API_KEY = True

# ============================================================
# BUILDING PARAMETERS
# ============================================================

DEFAULT_BUILDING_HEIGHT = 6.0  # meters
LEVEL_HEIGHT = 3.0             # meters per floor

# ============================================================
# ROAD PARAMETERS
# ============================================================

ROAD_HIGHWAY_TYPES = [
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

ROAD_CLASS_CODE = {
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "tertiary": 5,
    "unclassified": 6,
    "residential": 7,
    "service": 8,
    "living_street": 9,
    "track": 10,
    "path": 11,
    "footway": 12,
    "cycleway": 13,
    "pedestrian": 14,
    "other": 99,
}

# ============================================================
# TERRAIN PARAMETERS
# ============================================================

TRI_WINDOW_SIZE = 3


# ============================================================
# GEOMETRY HELPERS
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


def get_bbox_from_polygon(poly_gdf, padding_deg=0.0):
    """
    Return bbox as:
        west, south, east, north
    """
    west, south, east, north = poly_gdf.total_bounds

    west -= padding_deg
    south -= padding_deg
    east += padding_deg
    north += padding_deg

    return west, south, east, north


# ============================================================
# OPENTOPOGRAPHY DEM DOWNLOAD
# ============================================================

def check_opentopography_api_key():
    """
    Check that API key is set.
    """
    bad_values = [
        "",
        "PUT_YOUR_OPENTOPOGRAPHY_API_KEY_HERE",
        "YOUR_API_KEY_HERE",
        None,
    ]

    if REQUIRE_API_KEY and OPENTOPOGRAPHY_API_KEY in bad_values:
        raise ValueError(
            "\n[ERROR] OpenTopography API key is not set.\n\n"
            "Please edit the script and set:\n\n"
            "    OPENTOPOGRAPHY_API_KEY = \"your_key_here\"\n\n"
            "Get an API key from:\n"
            "    https://opentopography.org/\n"
        )


def download_opentopography_dem(
    west,
    south,
    east,
    north,
    out_tif,
    demtype="SRTMGL1",
    api_key=None,
):
    """
    Download DEM from OpenTopography Global DEM API.

    API endpoint:
        https://portal.opentopography.org/API/globaldem

    Parameters:
        demtype:
            SRTMGL1, SRTMGL3, AW3D30, COP30, COP90, NASADEM, etc.
    """

    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    url = "https://portal.opentopography.org/API/globaldem"

    params = {
        "demtype": demtype,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
    }

    if api_key is not None and str(api_key).strip() != "":
        params["API_Key"] = api_key

    print("\n[INFO] Downloading DEM from OpenTopography")
    print(f"[INFO] DEMTYPE: {demtype}")
    print(f"[INFO] BBOX:")
    print(f"       west  = {west}")
    print(f"       south = {south}")
    print(f"       east  = {east}")
    print(f"       north = {north}")

    response = requests.get(url, params=params, timeout=240)

    if response.status_code != 200:
        raise RuntimeError(
            "\n[ERROR] OpenTopography DEM download failed.\n"
            f"Status code: {response.status_code}\n"
            f"Response:\n{response.text[:2000]}\n\n"
            "Common fixes:\n"
            "1. Check OPENTOPOGRAPHY_API_KEY.\n"
            "2. Try another DEMTYPE, e.g., SRTMGL1 or COP30.\n"
            "3. Check your bbox values.\n"
        )

    content_type = response.headers.get("Content-Type", "")

    if "xml" in content_type.lower() or response.content[:20].startswith(b"<?xml"):
        raise RuntimeError(
            "\n[ERROR] OpenTopography returned XML/error instead of GeoTIFF.\n"
            f"Content-Type: {content_type}\n"
            f"Response:\n{response.text[:2000]}"
        )

    with open(out_tif, "wb") as f:
        f.write(response.content)

    if out_tif.stat().st_size < 1000:
        raise RuntimeError(
            f"[ERROR] Downloaded DEM file is too small: {out_tif}"
        )

    print(f"[OK] Saved DEM from OpenTopography: {out_tif}")


# ============================================================
# RASTER CLIPPING AND XYZ
# ============================================================

def clip_raster_by_polygon(in_tif, poly_gdf, out_tif):
    """
    Clip raster by polygon.
    """
    in_tif = Path(in_tif)
    out_tif = Path(out_tif)

    with rasterio.open(in_tif) as src:
        poly_for_raster = poly_gdf.to_crs(src.crs)
        geoms = [geom for geom in poly_for_raster.geometry]

        nodata_value = src.nodata
        if nodata_value is None:
            nodata_value = -9999.0

        clipped, clipped_transform = mask(
            src,
            geoms,
            crop=True,
            nodata=nodata_value,
            filled=True,
        )

        profile = src.profile.copy()
        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=clipped_transform,
            nodata=nodata_value,
            compress="lzw",
        )

        out_tif.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(clipped)

    print(f"[OK] Saved clipped DEM: {out_tif}")


def raster_to_xyz(in_tif, out_xyz, band=1):
    """
    Convert raster to XYZ:
        lon lat value
    """
    in_tif = Path(in_tif)
    out_xyz = Path(out_xyz)

    with rasterio.open(in_tif) as src:
        arr = src.read(band).astype(float)
        nodata = src.nodata

        if nodata is not None:
            arr[arr == nodata] = np.nan

        rows, cols = np.where(np.isfinite(arr))

        if len(rows) == 0:
            out_xyz.write_text("")
            print(f"[WARN] No valid raster cells. Empty XYZ saved: {out_xyz}")
            return

        xs, ys = xy(src.transform, rows, cols, offset="center")
        vals = arr[rows, cols]

        points = gpd.GeoDataFrame(
            {"value": vals},
            geometry=gpd.points_from_xy(xs, ys),
            crs=src.crs,
        ).to_crs("EPSG:4326")

    df = pd.DataFrame({
        "lon": points.geometry.x,
        "lat": points.geometry.y,
        "value": points["value"].to_numpy(),
    })

    df.to_csv(
        out_xyz,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    print(f"[OK] Saved XYZ: {out_xyz}")


def calculate_slope_and_tri(dem_tif, out_slope_tif, out_tri_tif):
    """
    Calculate:
        slope in degrees
        TRI in meters
    """
    dem_tif = Path(dem_tif)

    with rasterio.open(dem_tif) as src:
        dem = src.read(1).astype(float)
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata

        if nodata is not None:
            dem[dem == nodata] = np.nan

        if src.crs and src.crs.is_geographic:
            center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
            dy_m = abs(transform.e) * 111_320.0
            dx_m = abs(transform.a) * 111_320.0 * np.cos(np.deg2rad(center_lat))
        else:
            dx_m = abs(transform.a)
            dy_m = abs(transform.e)

        dz_dy, dz_dx = np.gradient(dem, dy_m, dx_m)

        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.rad2deg(slope_rad)

        def tri_func(window):
            center = window[len(window) // 2]

            if not np.isfinite(center):
                return np.nan

            diff = window - center
            return np.sqrt(np.nanmean(diff**2))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tri = generic_filter(
                dem,
                tri_func,
                size=TRI_WINDOW_SIZE,
                mode="nearest",
            )

        profile.update(
            dtype="float32",
            nodata=-9999.0,
            count=1,
            compress="lzw",
        )

        slope_write = np.where(
            np.isfinite(slope_deg),
            slope_deg,
            -9999.0,
        ).astype("float32")

        tri_write = np.where(
            np.isfinite(tri),
            tri,
            -9999.0,
        ).astype("float32")

        with rasterio.open(out_slope_tif, "w", **profile) as dst:
            dst.write(slope_write, 1)

        with rasterio.open(out_tri_tif, "w", **profile) as dst:
            dst.write(tri_write, 1)

    print(f"[OK] Saved slope raster: {out_slope_tif}")
    print(f"[OK] Saved TRI raster: {out_tri_tif}")


# ============================================================
# BUILDING FUNCTIONS
# ============================================================

def parse_building_height(row):
    """
    Estimate building height in meters.
    """
    for key in ["height", "building:height"]:
        if key in row and pd.notna(row[key]):
            raw = str(row[key]).lower()
            raw = raw.replace("meters", "")
            raw = raw.replace("meter", "")
            raw = raw.replace("m", "")
            raw = raw.strip()

            try:
                return float(raw)
            except ValueError:
                pass

    if "building:levels" in row and pd.notna(row["building:levels"]):
        raw = str(row["building:levels"]).strip()

        try:
            return float(raw) * LEVEL_HEIGHT
        except ValueError:
            pass

    return DEFAULT_BUILDING_HEIGHT


def write_empty_gpkg(out_gpkg, columns=None, crs="EPSG:4326"):
    """
    Write empty GeoPackage.
    """
    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    data = {}

    if columns:
        for name, dtype in columns.items():
            data[name] = pd.Series(dtype=dtype)

    empty = gpd.GeoDataFrame(
        data,
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )

    empty.to_file(out_gpkg, driver="GPKG")
    return empty


def download_osm_buildings_bbox(west, south, east, north, out_gpkg):
    """
    Download OSM buildings inside bbox.
    """
    print("\n[INFO] Downloading OSM buildings from bbox")

    tags = {"building": True}

    try:
        buildings = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=tags,
        )
    except TypeError:
        try:
            buildings = ox.features_from_bbox(
                (west, south, east, north),
                tags=tags,
            )
        except TypeError:
            buildings = ox.features_from_bbox(
                north,
                south,
                east,
                west,
                tags,
            )

    if buildings.empty:
        print("[WARN] No OSM buildings found.")
        return write_empty_gpkg(
            out_gpkg,
            columns={"height_m": "float"},
        )

    buildings = buildings[
        buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if buildings.empty:
        print("[WARN] No building polygons found.")
        return write_empty_gpkg(
            out_gpkg,
            columns={"height_m": "float"},
        )

    buildings = buildings.to_crs("EPSG:4326").reset_index(drop=True)
    buildings["height_m"] = buildings.apply(parse_building_height, axis=1)

    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    buildings.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved bbox buildings: {out_gpkg}")
    print(f"[INFO] Number of bbox buildings: {len(buildings)}")

    return buildings


def clip_buildings_by_polygon(buildings_gdf, poly_gdf, out_gpkg):
    """
    Clip buildings by polygon.
    """
    out_gpkg = Path(out_gpkg)

    if buildings_gdf.empty:
        print("[WARN] No buildings to clip.")
        return write_empty_gpkg(
            out_gpkg,
            columns={"height_m": "float"},
        )

    buildings = buildings_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)
    polygon = poly_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)

    clipped = gpd.clip(buildings, polygon)

    if clipped.empty:
        print("[WARN] No buildings inside polygon.")
        return write_empty_gpkg(
            out_gpkg,
            columns={"height_m": "float"},
        )

    clipped = clipped.copy().reset_index(drop=True)

    if "height_m" not in clipped.columns:
        clipped["height_m"] = DEFAULT_BUILDING_HEIGHT

    clipped.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved clipped buildings: {out_gpkg}")
    print(f"[INFO] Number of clipped buildings: {len(clipped)}")

    return clipped


def save_building_centroids_xyz(gdf, out_xyz):
    """
    Save one point per building:
        lon lat building_height_m
    """
    out_xyz = Path(out_xyz)

    if gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] Empty building centroid XYZ saved: {out_xyz}")
        return

    gdf = gdf.copy().reset_index(drop=True)
    gdf = gdf[
        gdf.geometry.notna()
        & (~gdf.geometry.is_empty)
        & gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy().reset_index(drop=True)

    if gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] No valid building polygons. Empty XYZ saved: {out_xyz}")
        return

    projected_crs = gdf.estimate_utm_crs()
    gdf_projected = gdf.to_crs(projected_crs).reset_index(drop=True)

    centroid_geom = gdf_projected.geometry.centroid

    centroid_gdf = gpd.GeoDataFrame(
        {"height_m": gdf_projected["height_m"].to_numpy()},
        geometry=gpd.GeoSeries(
            centroid_geom.to_numpy(),
            crs=projected_crs,
        ),
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

    print(f"[OK] Saved building centroid XYZ: {out_xyz}")


def save_building_vertices_xyz(gdf, out_xyz):
    """
    Save building polygon vertices:
        lon lat building_height_m
    """
    out_xyz = Path(out_xyz)

    if gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] Empty building vertices XYZ saved: {out_xyz}")
        return

    gdf = gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)

    records = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        height = row.get("height_m", DEFAULT_BUILDING_HEIGHT)

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

    print(f"[OK] Saved building vertices XYZ: {out_xyz}")


def rasterize_buildings_to_dem_grid(gdf, dem_tif, out_xyz):
    """
    Rasterize building height to the same grid as DEM.
    """
    out_xyz = Path(out_xyz)

    with rasterio.open(dem_tif) as src:
        shape = (src.height, src.width)
        transform = src.transform

        if gdf.empty:
            building_grid = np.zeros(shape, dtype="float32")
        else:
            gdf_dem = gdf.to_crs(src.crs)

            shapes = [
                (geom, float(height))
                for geom, height in zip(gdf_dem.geometry, gdf_dem["height_m"])
                if geom is not None and not geom.is_empty
            ]

            building_grid = rasterize(
                shapes=shapes,
                out_shape=shape,
                transform=transform,
                fill=0.0,
                dtype="float32",
                all_touched=True,
            )

        rows, cols = np.where(np.isfinite(building_grid))
        xs, ys = xy(transform, rows, cols, offset="center")
        vals = building_grid[rows, cols]

        points = gpd.GeoDataFrame(
            {"height_m": vals},
            geometry=gpd.points_from_xy(xs, ys),
            crs=src.crs,
        ).to_crs("EPSG:4326")

    df = pd.DataFrame({
        "lon": points.geometry.x,
        "lat": points.geometry.y,
        "height_m": points["height_m"].to_numpy(),
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
# ROAD FUNCTIONS
# ============================================================

def normalize_highway_type(value):
    """
    Normalize OSM highway tag.
    """
    if value is None:
        return "other"

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "other"
        value = value[0]

    value = str(value)

    if value in ROAD_CLASS_CODE:
        return value

    return "other"


def download_osm_roads_bbox(west, south, east, north, out_gpkg):
    """
    Download OSM roads inside bbox.
    """
    print("\n[INFO] Downloading OSM roads from bbox")

    tags = {"highway": ROAD_HIGHWAY_TYPES}

    try:
        roads = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=tags,
        )
    except TypeError:
        try:
            roads = ox.features_from_bbox(
                (west, south, east, north),
                tags=tags,
            )
        except TypeError:
            roads = ox.features_from_bbox(
                north,
                south,
                east,
                west,
                tags,
            )

    if roads.empty:
        print("[WARN] No OSM roads found.")
        return write_empty_gpkg(
            out_gpkg,
            columns={
                "highway": "str",
                "highway_simple": "str",
                "road_code": "int",
            },
        )

    roads = roads[
        roads.geometry.type.isin(["LineString", "MultiLineString"])
    ].copy()

    if roads.empty:
        print("[WARN] No road lines found.")
        return write_empty_gpkg(
            out_gpkg,
            columns={
                "highway": "str",
                "highway_simple": "str",
                "road_code": "int",
            },
        )

    roads = roads.to_crs("EPSG:4326").reset_index(drop=True)

    if "highway" not in roads.columns:
        roads["highway"] = "other"

    roads["highway_simple"] = roads["highway"].apply(normalize_highway_type)
    roads["road_code"] = roads["highway_simple"].map(ROAD_CLASS_CODE).fillna(99).astype(int)

    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    roads.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved bbox roads: {out_gpkg}")
    print(f"[INFO] Number of bbox roads: {len(roads)}")
    print("[INFO] Road classes:")
    print(roads["highway_simple"].value_counts())

    return roads


def clip_roads_by_polygon(roads_gdf, poly_gdf, out_gpkg):
    """
    Clip roads by polygon.
    """
    out_gpkg = Path(out_gpkg)

    if roads_gdf.empty:
        print("[WARN] No roads to clip.")
        return write_empty_gpkg(
            out_gpkg,
            columns={
                "highway": "str",
                "highway_simple": "str",
                "road_code": "int",
            },
        )

    roads = roads_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)
    polygon = poly_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)

    clipped = gpd.clip(roads, polygon)

    if clipped.empty:
        print("[WARN] No roads inside polygon.")
        return write_empty_gpkg(
            out_gpkg,
            columns={
                "highway": "str",
                "highway_simple": "str",
                "road_code": "int",
            },
        )

    clipped = clipped.copy().reset_index(drop=True)

    if "highway_simple" not in clipped.columns:
        clipped["highway_simple"] = clipped["highway"].apply(normalize_highway_type)

    if "road_code" not in clipped.columns:
        clipped["road_code"] = clipped["highway_simple"].map(ROAD_CLASS_CODE).fillna(99).astype(int)

    clipped.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved clipped roads: {out_gpkg}")
    print(f"[INFO] Number of clipped roads: {len(clipped)}")
    print("[INFO] Clipped road classes:")
    print(clipped["highway_simple"].value_counts())

    return clipped


def save_road_vertices_xyz(roads_gdf, out_xyz):
    """
    Save road centerline vertices:
        lon lat road_code segment_id
    """
    out_xyz = Path(out_xyz)

    if roads_gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] Empty road XYZ saved: {out_xyz}")
        return

    roads = roads_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)

    records = []
    segment_id = 0

    for _, row in roads.iterrows():
        geom = row.geometry
        road_code = int(row.get("road_code", 99))

        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            for x, y in line.coords:
                records.append((x, y, road_code, segment_id))
            segment_id += 1

    df = pd.DataFrame(
        records,
        columns=["lon", "lat", "road_code", "segment_id"],
    )

    df.to_csv(
        out_xyz,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    print(f"[OK] Saved road vertices XYZ: {out_xyz}")
    print("     Format: lon lat road_code segment_id")


# ============================================================
# MAIN
# ============================================================

def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n========== HOA LAC OPENTOPOGRAPHY DOWNLOAD ==========")

    check_opentopography_api_key()

    # --------------------------------------------------------
    # 1. Polygon and bbox
    # --------------------------------------------------------
    poly_gdf = make_hoalac_polygon_gdf()

    poly_file = outdir / "hoalac_polygon.gpkg"
    poly_gdf.to_file(poly_file, driver="GPKG")
    print(f"[OK] Saved Hoa Lac polygon: {poly_file}")

    west, south, east, north = get_bbox_from_polygon(
        poly_gdf,
        padding_deg=BBOX_PADDING_DEG,
    )

    print("\n[INFO] Download bbox:")
    print(f"  WEST  = {west}")
    print(f"  SOUTH = {south}")
    print(f"  EAST  = {east}")
    print(f"  NORTH = {north}")

    # --------------------------------------------------------
    # 2. Download OpenTopography DEM
    # --------------------------------------------------------
    dem_bbox_tif = outdir / "dem_opentopography_bbox.tif"
    dem_clip_tif = outdir / "dem_hoalac_clipped.tif"

    download_opentopography_dem(
        west=west,
        south=south,
        east=east,
        north=north,
        out_tif=dem_bbox_tif,
        demtype=DEMTYPE,
        api_key=OPENTOPOGRAPHY_API_KEY,
    )

    # --------------------------------------------------------
    # 3. Clip DEM and export DEM XYZ
    # --------------------------------------------------------
    clip_raster_by_polygon(
        in_tif=dem_bbox_tif,
        poly_gdf=poly_gdf,
        out_tif=dem_clip_tif,
    )

    raster_to_xyz(
        in_tif=dem_clip_tif,
        out_xyz=outdir / "terrain_dem_hoalac.xyz",
    )

    # --------------------------------------------------------
    # 4. Slope and TRI
    # --------------------------------------------------------
    slope_tif = outdir / "terrain_slope_hoalac.tif"
    tri_tif = outdir / "terrain_ruggedness_TRI_hoalac.tif"

    calculate_slope_and_tri(
        dem_tif=dem_clip_tif,
        out_slope_tif=slope_tif,
        out_tri_tif=tri_tif,
    )

    raster_to_xyz(
        in_tif=slope_tif,
        out_xyz=outdir / "terrain_slope_hoalac.xyz",
    )

    raster_to_xyz(
        in_tif=tri_tif,
        out_xyz=outdir / "terrain_ruggedness_TRI_hoalac.xyz",
    )

    # --------------------------------------------------------
    # 5. Buildings
    # --------------------------------------------------------
    buildings_bbox = download_osm_buildings_bbox(
        west=west,
        south=south,
        east=east,
        north=north,
        out_gpkg=outdir / "buildings_bbox.gpkg",
    )

    buildings_clip = clip_buildings_by_polygon(
        buildings_gdf=buildings_bbox,
        poly_gdf=poly_gdf,
        out_gpkg=outdir / "buildings_hoalac_clipped.gpkg",
    )

    save_building_centroids_xyz(
        buildings_clip,
        outdir / "buildings_centroid_hoalac.xyz",
    )

    save_building_vertices_xyz(
        buildings_clip,
        outdir / "buildings_vertices_hoalac.xyz",
    )

    rasterize_buildings_to_dem_grid(
        buildings_clip,
        dem_tif=dem_clip_tif,
        out_xyz=outdir / "buildings_grid_hoalac.xyz",
    )

    # --------------------------------------------------------
    # 6. Roads
    # --------------------------------------------------------
    roads_bbox = download_osm_roads_bbox(
        west=west,
        south=south,
        east=east,
        north=north,
        out_gpkg=outdir / "roads_bbox.gpkg",
    )

    roads_clip = clip_roads_by_polygon(
        roads_gdf=roads_bbox,
        poly_gdf=poly_gdf,
        out_gpkg=outdir / "roads_hoalac_clipped.gpkg",
    )

    save_road_vertices_xyz(
        roads_clip,
        outdir / "roads_vertices_hoalac.xyz",
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------
    print("\n========== DONE ==========")
    print(f"All output saved in: {outdir.resolve()}")

    print("\nImportant files:")
    print(f"  DEM GeoTIFF:          {outdir / 'dem_hoalac_clipped.tif'}")
    print(f"  DEM XYZ:              {outdir / 'terrain_dem_hoalac.xyz'}")
    print(f"  Slope XYZ:            {outdir / 'terrain_slope_hoalac.xyz'}")
    print(f"  TRI XYZ:              {outdir / 'terrain_ruggedness_TRI_hoalac.xyz'}")
    print(f"  Buildings GPKG:       {outdir / 'buildings_hoalac_clipped.gpkg'}")
    print(f"  Building grid XYZ:    {outdir / 'buildings_grid_hoalac.xyz'}")
    print(f"  Roads GPKG:           {outdir / 'roads_hoalac_clipped.gpkg'}")
    print(f"  Road vertices XYZ:    {outdir / 'roads_vertices_hoalac.xyz'}")


if __name__ == "__main__":
    main()
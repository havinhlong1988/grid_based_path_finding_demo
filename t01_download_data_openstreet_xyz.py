#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download DEM/topography and OSM building data inside Hoa Lac polygon.

This version does NOT use:
    - OpenTopography API
    - elevation package
    - make command

DEM source:
    AWS Terrain Tiles / Mapzen GeoTIFF tiles
    https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif

Building source:
    OpenStreetMap through OSMnx

Outputs:
    output_hoalac_hitech_park/
    ├── hoalac_polygon.gpkg
    ├── dem_tiles/
    ├── dem_bbox_merged.tif
    ├── dem_hoalac_clipped.tif
    ├── terrain_dem_hoalac.xyz
    ├── terrain_slope_hoalac.xyz
    ├── terrain_ruggedness_TRI_hoalac.xyz
    ├── buildings_bbox.gpkg
    ├── buildings_hoalac_clipped.gpkg
    ├── buildings_centroid_hoalac.xyz
    ├── buildings_vertices_hoalac.xyz
    └── buildings_grid_hoalac.xyz

XYZ format:
    lon lat value
"""

from pathlib import Path
import math
import warnings
import requests

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.transform import xy
from shapely.geometry import Polygon
from scipy.ndimage import generic_filter


# ============================================================
# USER INPUT PARAMETERS
# ============================================================

# Hoa Lac polygon, format: lon, lat
# This polygon is used for final clipping.
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
OUTDIR = "output_hoalac_hitech_park"

# DEM tile zoom level.
# Higher zoom = finer grid, more tiles.
# Recommended:
#   12 = coarser, faster
#   13 = good for small UAV/UTM area
#   14 = finer, more files
DEM_ZOOM = 13

# Extra padding around polygon bbox for downloading DEM/buildings.
# 0.002 degree is roughly 200 m.
BBOX_PADDING_DEG = 0.002

# OSM building height assumptions.
# OSM often has footprints but no height.
DEFAULT_BUILDING_HEIGHT = 6.0  # meter
LEVEL_HEIGHT = 3.0             # meter per floor

# Terrain ruggedness window
TRI_WINDOW_SIZE = 3

# Road data from OpenStreetMap
# Common highway classes:
# motorway, trunk, primary, secondary, tertiary,
# unclassified, residential, service, living_street, track, path
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
]

# Save road vertices with road class as integer code
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
    "other": 99,
}

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
    Return bbox from polygon as:
        west, south, east, north
    """
    west, south, east, north = poly_gdf.total_bounds

    west -= padding_deg
    south -= padding_deg
    east += padding_deg
    north += padding_deg

    return west, south, east, north


# ============================================================
# AWS TERRAIN TILE HELPERS
# ============================================================

def lonlat_to_tile(lon, lat, zoom):
    """
    Convert lon/lat to Web Mercator tile x/y at zoom.
    Standard slippy-map tiling.
    """
    lat = max(min(lat, 85.05112878), -85.05112878)

    lat_rad = math.radians(lat)
    n = 2 ** zoom

    x = int((lon + 180.0) / 360.0 * n)
    y = int(
        (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    )

    x = max(0, min(x, n - 1))
    y = max(0, min(y, n - 1))

    return x, y


def get_tile_range_from_bbox(west, south, east, north, zoom):
    """
    Get all terrain tiles covering bbox.
    """
    x_min, y_north = lonlat_to_tile(west, north, zoom)
    x_max, y_south = lonlat_to_tile(east, south, zoom)

    x0 = min(x_min, x_max)
    x1 = max(x_min, x_max)
    y0 = min(y_north, y_south)
    y1 = max(y_north, y_south)

    return x0, x1, y0, y1


def download_aws_terrain_tiles(
    west,
    south,
    east,
    north,
    zoom,
    tile_dir,
):
    """
    Download AWS Terrain Tiles GeoTIFF tiles covering bbox.
    """
    tile_dir = Path(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)

    x0, x1, y0, y1 = get_tile_range_from_bbox(
        west=west,
        south=south,
        east=east,
        north=north,
        zoom=zoom,
    )

    print("\n[INFO] Downloading DEM tiles from AWS Terrain Tiles")
    print(f"[INFO] Zoom: {zoom}")
    print(f"[INFO] Tile x range: {x0} to {x1}")
    print(f"[INFO] Tile y range: {y0} to {y1}")

    downloaded_tiles = []

    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{zoom}/{x}/{y}.tif"
            out_file = tile_dir / f"terrain_z{zoom}_x{x}_y{y}.tif"

            if out_file.exists() and out_file.stat().st_size > 0:
                print(f"[SKIP] Existing tile: {out_file.name}")
                downloaded_tiles.append(out_file)
                continue

            print(f"[INFO] Downloading tile z={zoom}, x={x}, y={y}")

            try:
                r = requests.get(url, timeout=120)
            except requests.RequestException as e:
                print(f"[WARN] Request failed for tile {x}/{y}: {e}")
                continue

            if r.status_code != 200:
                print(f"[WARN] Tile not available: {url}")
                print(f"[WARN] Status code: {r.status_code}")
                continue

            with open(out_file, "wb") as f:
                f.write(r.content)

            if out_file.stat().st_size == 0:
                print(f"[WARN] Empty tile: {out_file}")
                continue

            downloaded_tiles.append(out_file)
            print(f"[OK] Saved tile: {out_file}")

    if len(downloaded_tiles) == 0:
        raise RuntimeError(
            "No DEM tiles were downloaded. "
            "Check internet connection or try a lower DEM_ZOOM, e.g. DEM_ZOOM = 12."
        )

    return downloaded_tiles


def mosaic_dem_tiles(tile_files, out_tif):
    """
    Merge downloaded GeoTIFF DEM tiles into one raster.
    """
    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Merging DEM tiles")

    datasets = []

    try:
        for f in tile_files:
            datasets.append(rasterio.open(f))

        mosaic_arr, mosaic_transform = merge(datasets)

        profile = datasets[0].profile.copy()
        profile.update(
            height=mosaic_arr.shape[1],
            width=mosaic_arr.shape[2],
            transform=mosaic_transform,
            compress="lzw",
            nodata=datasets[0].nodata,
        )

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(mosaic_arr)

    finally:
        for ds in datasets:
            ds.close()

    print(f"[OK] Saved merged DEM: {out_tif}")


def download_dem_aws_terrain(
    west,
    south,
    east,
    north,
    out_tif,
    tile_dir,
    zoom=13,
):
    """
    Main DEM downloader using AWS Terrain Tiles.
    """
    tile_files = download_aws_terrain_tiles(
        west=west,
        south=south,
        east=east,
        north=north,
        zoom=zoom,
        tile_dir=tile_dir,
    )

    mosaic_dem_tiles(
        tile_files=tile_files,
        out_tif=out_tif,
    )


# ============================================================
# RASTER CLIPPING
# ============================================================

def clip_raster_by_polygon(in_tif, poly_gdf, out_tif):
    """
    Clip raster by polygon boundary.
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

    print(f"[OK] Saved clipped raster: {out_tif}")


# ============================================================
# RASTER TO XYZ
# ============================================================

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


# ============================================================
# TERRAIN DERIVATIVES
# ============================================================

def calculate_slope_and_tri(dem_tif, out_slope_tif, out_tri_tif):
    """
    Calculate:
        slope in degrees
        TRI, terrain ruggedness index, in meters
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
# OSM BUILDING DOWNLOAD
# ============================================================

def parse_building_height(row):
    """
    Estimate building height in meters.

    Priority:
        1. height
        2. building:height
        3. building:levels * LEVEL_HEIGHT
        4. DEFAULT_BUILDING_HEIGHT
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


def write_empty_gpkg(out_gpkg, crs="EPSG:4326"):
    """
    Write empty GeoPackage with minimal schema.
    """
    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    empty = gpd.GeoDataFrame(
        {"height_m": pd.Series(dtype="float")},
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )

    empty.to_file(out_gpkg, driver="GPKG")
    return empty


def download_osm_buildings_bbox(west, south, east, north, out_gpkg):
    """
    Download OSM building footprints inside bbox.
    """
    print("\n[INFO] Downloading OSM buildings from bbox")

    tags = {"building": True}

    try:
        # OSMnx 2.x
        gdf = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=tags,
        )
    except TypeError:
        try:
            # Some versions accept positional bbox
            gdf = ox.features_from_bbox(
                (west, south, east, north),
                tags=tags,
            )
        except TypeError:
            # Older OSMnx style
            gdf = ox.features_from_bbox(
                north,
                south,
                east,
                west,
                tags,
            )

    if gdf.empty:
        print("[WARN] No OSM buildings found in bbox.")
        return write_empty_gpkg(out_gpkg)

    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    if gdf.empty:
        print("[WARN] OSM data found, but no building polygons.")
        return write_empty_gpkg(out_gpkg)

    gdf = gdf.to_crs("EPSG:4326")
    gdf["height_m"] = gdf.apply(parse_building_height, axis=1)

    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved bbox buildings: {out_gpkg}")
    print(f"[INFO] Number of bbox buildings: {len(gdf)}")

    return gdf


def clip_buildings_by_polygon(buildings_gdf, poly_gdf, out_gpkg):
    """
    Clip OSM buildings by Hoa Lac polygon.
    """
    out_gpkg = Path(out_gpkg)

    if buildings_gdf.empty:
        print("[WARN] No buildings to clip.")
        return write_empty_gpkg(out_gpkg)

    buildings = buildings_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)
    polygon = poly_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)

    clipped = gpd.clip(buildings, polygon)

    if clipped.empty:
        print("[WARN] No buildings inside Hoa Lac polygon.")
        return write_empty_gpkg(out_gpkg)

    # Important fix for OSMnx MultiIndex problem
    clipped = clipped.copy().reset_index(drop=True)

    if "height_m" not in clipped.columns:
        clipped["height_m"] = DEFAULT_BUILDING_HEIGHT

    clipped.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved clipped buildings: {out_gpkg}")
    print(f"[INFO] Number of clipped buildings: {len(clipped)}")

    return clipped


# ============================================================
# BUILDING XYZ EXPORT
# ============================================================

def save_building_centroids_xyz(gdf, out_xyz):
    """
    Save one point per building:
        lon lat building_height_m

    Fixed for OSMnx MultiIndex / clipped GeoDataFrame index problem.
    """
    out_xyz = Path(out_xyz)

    if gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] Empty building centroid XYZ saved: {out_xyz}")
        return

    # Important fix:
    # OSMnx often returns MultiIndex. After clipping, index can be incompatible.
    # Reset index to simple 0,1,2,...
    gdf = gdf.copy().reset_index(drop=True)

    # Keep only valid geometries
    gdf = gdf[
        gdf.geometry.notna()
        & (~gdf.geometry.is_empty)
        & gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy().reset_index(drop=True)

    if gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] No valid building polygons. Empty XYZ saved: {out_xyz}")
        return

    if "height_m" not in gdf.columns:
        gdf["height_m"] = DEFAULT_BUILDING_HEIGHT

    # Calculate centroid in projected CRS, then convert back to lon/lat
    projected_crs = gdf.estimate_utm_crs()
    gdf_projected = gdf.to_crs(projected_crs).reset_index(drop=True)

    centroid_geom = gdf_projected.geometry.centroid

    centroid_gdf = gpd.GeoDataFrame(
        {
            "height_m": gdf_projected["height_m"].to_numpy()
        },
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

    gdf = gdf.to_crs("EPSG:4326")

    records = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        height = row["height_m"]

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
    Rasterize building height to the same grid as clipped DEM.

    Output:
        lon lat building_height_m

    No-building cells are 0.
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
# OSM ROAD DOWNLOAD
# ============================================================

def normalize_highway_type(value):
    """
    Normalize OSM highway tag.

    OSM highway can be:
        string, list, tuple, or missing
    """
    if value is None or pd.isna(value):
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
    Download OSM road centerlines inside bbox.

    Output GeoPackage contains LineString/MultiLineString roads.
    """

    print("\n[INFO] Downloading OSM roads from bbox")

    tags = {
        "highway": ROAD_HIGHWAY_TYPES
    }

    try:
        # OSMnx 2.x
        roads = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=tags,
        )
    except TypeError:
        try:
            # Some OSMnx versions
            roads = ox.features_from_bbox(
                (west, south, east, north),
                tags=tags,
            )
        except TypeError:
            # Older OSMnx style
            roads = ox.features_from_bbox(
                north,
                south,
                east,
                west,
                tags,
            )

    if roads.empty:
        print("[WARN] No OSM roads found in bbox.")
        empty = gpd.GeoDataFrame(
            {
                "highway": pd.Series(dtype="str"),
                "road_code": pd.Series(dtype="int"),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
        empty.to_file(out_gpkg, driver="GPKG")
        return empty

    roads = roads[
        roads.geometry.type.isin(
            ["LineString", "MultiLineString"]
        )
    ].copy()

    if roads.empty:
        print("[WARN] OSM data found, but no road lines.")
        empty = gpd.GeoDataFrame(
            {
                "highway": pd.Series(dtype="str"),
                "road_code": pd.Series(dtype="int"),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
        empty.to_file(out_gpkg, driver="GPKG")
        return empty

    roads = roads.to_crs("EPSG:4326").reset_index(drop=True)

    roads["highway_simple"] = roads["highway"].apply(normalize_highway_type)
    roads["road_code"] = roads["highway_simple"].map(ROAD_CLASS_CODE).fillna(99).astype(int)

    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    roads.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved bbox roads: {out_gpkg}")
    print(f"[INFO] Number of bbox road segments: {len(roads)}")

    return roads


def clip_roads_by_polygon(roads_gdf, poly_gdf, out_gpkg):
    """
    Clip OSM roads by Hoa Lac polygon.
    """

    out_gpkg = Path(out_gpkg)

    if roads_gdf.empty:
        print("[WARN] No roads to clip.")
        empty = gpd.GeoDataFrame(
            {
                "highway": pd.Series(dtype="str"),
                "road_code": pd.Series(dtype="int"),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
        empty.to_file(out_gpkg, driver="GPKG")
        return empty

    roads = roads_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)
    polygon = poly_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)

    clipped = gpd.clip(roads, polygon)

    if clipped.empty:
        print("[WARN] No roads inside Hoa Lac polygon.")
        empty = gpd.GeoDataFrame(
            {
                "highway": pd.Series(dtype="str"),
                "road_code": pd.Series(dtype="int"),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
        empty.to_file(out_gpkg, driver="GPKG")
        return empty

    clipped = clipped.copy().reset_index(drop=True)

    if "highway_simple" not in clipped.columns:
        clipped["highway_simple"] = clipped["highway"].apply(normalize_highway_type)

    if "road_code" not in clipped.columns:
        clipped["road_code"] = clipped["highway_simple"].map(ROAD_CLASS_CODE).fillna(99).astype(int)

    clipped.to_file(out_gpkg, driver="GPKG")

    print(f"[OK] Saved clipped roads: {out_gpkg}")
    print(f"[INFO] Number of clipped road segments: {len(clipped)}")

    return clipped


def save_road_vertices_xyz(roads_gdf, out_xyz):
    """
    Save road centerline vertices as XYZ.

    Format:
        lon lat road_code

    road_code:
        1  motorway
        2  trunk
        3  primary
        4  secondary
        5  tertiary
        6  unclassified
        7  residential
        8  service
        9  living_street
        10 track
        11 path
        99 other
    """

    out_xyz = Path(out_xyz)

    if roads_gdf.empty:
        out_xyz.write_text("")
        print(f"[WARN] Empty road XYZ saved: {out_xyz}")
        return

    roads = roads_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)

    records = []

    for _, row in roads.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        road_code = int(row.get("road_code", 99))

        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            for x, y in line.coords:
                records.append((x, y, road_code))

            # Add NaN separator between line segments.
            # Useful if later you want line plotting.
            records.append((np.nan, np.nan, np.nan))

    df = pd.DataFrame(records, columns=["lon", "lat", "road_code"])

    df.to_csv(
        out_xyz,
        sep=" ",
        index=False,
        header=False,
        float_format="%.8f",
    )

    print(f"[OK] Saved road vertices XYZ: {out_xyz}")

# ============================================================
# MAIN
# ============================================================

def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n========== HOA LAC DATA DOWNLOAD ==========")
    print("[INFO] DEM source: AWS Terrain Tiles / Mapzen GeoTIFF")
    print("[INFO] Building source: OpenStreetMap")

    # --------------------------------------------------------
    # 1. Prepare Hoa Lac polygon
    # --------------------------------------------------------
    poly_gdf = make_hoalac_polygon_gdf()

    poly_file = outdir / "hoalac_polygon.gpkg"
    poly_gdf.to_file(poly_file, driver="GPKG")
    print(f"[OK] Saved Hoa Lac polygon: {poly_file}")

    # --------------------------------------------------------
    # 2. Get bbox from polygon
    # --------------------------------------------------------
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
    # 3. Download DEM from AWS terrain tiles and merge
    # --------------------------------------------------------
    dem_tile_dir = outdir / "dem_tiles"
    dem_bbox_tif = outdir / "dem_bbox_merged.tif"
    dem_clip_tif = outdir / "dem_hoalac_clipped.tif"

    download_dem_aws_terrain(
        west=west,
        south=south,
        east=east,
        north=north,
        out_tif=dem_bbox_tif,
        tile_dir=dem_tile_dir,
        zoom=DEM_ZOOM,
    )

    # --------------------------------------------------------
    # 4. Clip DEM to Hoa Lac polygon
    # --------------------------------------------------------
    clip_raster_by_polygon(
        in_tif=dem_bbox_tif,
        poly_gdf=poly_gdf,
        out_tif=dem_clip_tif,
    )

    # --------------------------------------------------------
    # 5. Export DEM to XYZ
    # --------------------------------------------------------
    raster_to_xyz(
        in_tif=dem_clip_tif,
        out_xyz=outdir / "terrain_dem_hoalac.xyz",
    )

    # --------------------------------------------------------
    # 6. Calculate slope and TRI
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
    # 7. Download OSM buildings using bbox
    # --------------------------------------------------------
    buildings_bbox_gpkg = outdir / "buildings_bbox.gpkg"

    buildings_bbox = download_osm_buildings_bbox(
        west=west,
        south=south,
        east=east,
        north=north,
        out_gpkg=buildings_bbox_gpkg,
    )

    # --------------------------------------------------------
    # 8. Clip buildings to Hoa Lac polygon
    # --------------------------------------------------------
    buildings_clip_gpkg = outdir / "buildings_hoalac_clipped.gpkg"

    buildings_clip = clip_buildings_by_polygon(
        buildings_gdf=buildings_bbox,
        poly_gdf=poly_gdf,
        out_gpkg=buildings_clip_gpkg,
    )

    # --------------------------------------------------------
    # 9. Export buildings to XYZ
    # --------------------------------------------------------
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
    # 10. Download OSM roads using bbox
    # --------------------------------------------------------
    roads_bbox_gpkg = outdir / "roads_bbox.gpkg"

    roads_bbox = download_osm_roads_bbox(
        west=west,
        south=south,
        east=east,
        north=north,
        out_gpkg=roads_bbox_gpkg,
    )

    # --------------------------------------------------------
    # 11. Clip roads to Hoa Lac polygon
    # --------------------------------------------------------
    roads_clip_gpkg = outdir / "roads_hoalac_clipped.gpkg"

    roads_clip = clip_roads_by_polygon(
        roads_gdf=roads_bbox,
        poly_gdf=poly_gdf,
        out_gpkg=roads_clip_gpkg,
    )

    # --------------------------------------------------------
    # 12. Export road vertices to XYZ
    # --------------------------------------------------------
    save_road_vertices_xyz(
        roads_clip,
        outdir / "roads_vertices_hoalac.xyz",
    )
    
    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------
    print("\n========== DONE ==========")
    print(f"All output saved in: {outdir.resolve()}")

    print("\nImportant output files:")
    print(f"  Polygon:             {outdir / 'hoalac_polygon.gpkg'}")
    print(f"  DEM GeoTIFF:         {outdir / 'dem_hoalac_clipped.tif'}")
    print(f"  DEM XYZ:             {outdir / 'terrain_dem_hoalac.xyz'}")
    print(f"  Slope XYZ:           {outdir / 'terrain_slope_hoalac.xyz'}")
    print(f"  Ruggedness XYZ:      {outdir / 'terrain_ruggedness_TRI_hoalac.xyz'}")
    print(f"  Buildings GPKG:      {outdir / 'buildings_hoalac_clipped.gpkg'}")
    print(f"  Building centroid:   {outdir / 'buildings_centroid_hoalac.xyz'}")
    print(f"  Building vertices:   {outdir / 'buildings_vertices_hoalac.xyz'}")
    print(f"  Building grid:       {outdir / 'buildings_grid_hoalac.xyz'}")


if __name__ == "__main__":
    main()
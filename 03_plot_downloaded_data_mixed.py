#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Hoa Lac study-area outputs from the mixed download pipeline.

Input folder:
    output/01_HoaLac_studies_area/

Output folder:
    figures/01_HoaLac_studies_area/

Expected input files, in plotting order:
    metadata/study_area_aoi.gpkg
    osm/roads/osm_roads_edges.gpkg
    osm/roads/osm_road_class_summary.csv
    osm/extra_features/osm_extra_features.gpkg
    opentopography/opentopography_SRTMGL1_dem_wgs84.tif
    opentopography/opentopography_SRTMGL1_dem_utm.tif
    opentopography/terrain_products/slope_degree.tif
    opentopography/terrain_products/hillshade.tif
    openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg

Figures:
    00_overview_map.png
    01_study_area_aoi.png
    02_osm_roads_edges_by_class.png
    03_osm_road_class_summary.png
    04_osm_extra_features.png
    05_opentopography_SRTMGL1_dem_wgs84.png
    06_opentopography_SRTMGL1_dem_utm.png
    07_slope_degree.png
    08_hillshade.png
    09_obm_buildings_hoalac_clipped.png

Notes:
    - Uses PyGMT for map figures.
    - Uses matplotlib only for the road-class summary CSV bar chart.
    - Projected rasters such as UTM DEM/slope/hillshade are temporarily
      reprojected to EPSG:4326 before plotting with the lon/lat map frame.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt
import rasterio
import rasterio.mask
import rasterio.warp
from rasterio.enums import Resampling
from shapely.geometry import Polygon, mapping

import matplotlib.pyplot as plt


# ============================================================
# 0. USER INPUT PARAMETERS
# ============================================================

DATA_DIR = Path("output/01_HoaLac_studies_area")
FIG_DIR = Path("figures/01_HoaLac_studies_area")

TMP_DIR = FIG_DIR / "_tmp_reprojected_rasters"

# New output files from the download script
AOI_GPKG = DATA_DIR / "metadata/study_area_aoi.gpkg"
ROADS_GPKG = DATA_DIR / "osm/roads/osm_roads_edges.gpkg"
ROAD_SUMMARY_CSV = DATA_DIR / "osm/roads/osm_road_class_summary.csv"
OSM_EXTRA_GPKG = DATA_DIR / "osm/extra_features/osm_extra_features.gpkg"

DEM_WGS84_TIF = DATA_DIR / "opentopography/opentopography_SRTMGL1_dem_wgs84.tif"
DEM_UTM_TIF = DATA_DIR / "opentopography/opentopography_SRTMGL1_dem_utm.tif"
SLOPE_DEGREE_TIF = DATA_DIR / "opentopography/terrain_products/slope_degree.tif"
HILLSHADE_TIF = DATA_DIR / "opentopography/terrain_products/hillshade.tif"

OBM_BUILDINGS_GPKG = DATA_DIR / "openbuildingmap/clipped/obm_buildings_hoalac_clipped.gpkg"

# Hoa Lac polygon, lon/lat.
# Used as fallback if the AOI GPKG is missing.
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
REGION_PADDING = 0.003
DPI = 300

# Optional overlay switches
OVERLAY_ROADS_ON_RASTERS = True
OVERLAY_BUILDINGS_ON_RASTERS = False
OVERLAY_AOI_ON_ALL_MAPS = True


# ============================================================
# 1. MAP STYLE PARAMETERS
# ============================================================

POLYGON_PEN = "1.4p,purple"
ROAD_PEN = "0.45p,gray35"
BUILDING_FILL = "lightred@65"
BUILDING_PEN = "0.20p,black@35"
EXTRA_POLYGON_FILL = "gray80@65"
EXTRA_LINE_PEN = "0.45p,gray40"
EXTRA_POINT_STYLE = "c0.08c"
EXTRA_POINT_FILL = "black"

LEGEND_BOX_FILL = "white@10"
LEGEND_BOX_PEN = "0.5p,black"
LEGEND_POSITION = "JBL+jBL+o0.2c/0.2c"

ROAD_LEGEND_POSITION = "JBR+jBR+o0.2c/0.2c"
ROAD_LEGEND_BOX = "+gwhite@10+p0.5p,black"

# Road styles are adapted for both:
#   1. raw OSM highway classes
#   2. simplified classes from the download script:
#      expressway_or_trunk, service_or_track, non_motorized
ROAD_CLASS_STYLE = {
    "expressway_or_trunk": {"pen": "1.5p,red",       "label": "Expressway/trunk"},
    "motorway":            {"pen": "1.5p,red",       "label": "Motorway"},
    "trunk":               {"pen": "1.4p,orange",    "label": "Trunk"},
    "primary":             {"pen": "1.3p,yellow",    "label": "Primary"},
    "secondary":           {"pen": "1.1p,green",     "label": "Secondary"},
    "tertiary":            {"pen": "1.0p,cyan",      "label": "Tertiary"},
    "residential":         {"pen": "0.7p,blue",      "label": "Residential"},
    "service_or_track":    {"pen": "0.6p,gray55",    "label": "Service/track"},
    "service":             {"pen": "0.6p,gray55",    "label": "Service"},
    "track":               {"pen": "0.6p,brown",     "label": "Track"},
    "unclassified":        {"pen": "0.8p,gray35",    "label": "Unclassified"},
    "non_motorized":       {"pen": "0.5p,magenta",   "label": "Non-motorized"},
    "path":                {"pen": "0.5p,magenta",   "label": "Path"},
    "footway":             {"pen": "0.5p,magenta",   "label": "Footway"},
    "cycleway":            {"pen": "0.5p,darkgreen", "label": "Cycleway"},
    "pedestrian":          {"pen": "0.6p,darkgray",  "label": "Pedestrian"},
    "living_street":       {"pen": "0.6p,purple",    "label": "Living street"},
    "other":               {"pen": "0.5p,black",     "label": "Other"},
}

ROAD_PLOT_ORDER = [
    "expressway_or_trunk",
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service_or_track",
    "service",
    "track",
    "living_street",
    "non_motorized",
    "path",
    "footway",
    "cycleway",
    "pedestrian",
    "other",
]

# OSM extra feature styles.
# These are deliberately simple so the figure does not become too noisy.
EXTRA_TAG_PRIORITY = [
    "water",
    "waterway",
    "natural",
    "landuse",
    "aeroway",
    "railway",
    "amenity",
    "man_made",
    "leisure",
    "barrier",
    "building",
]

EXTRA_TAG_STYLE = {
    "water":    {"fill": "skyblue@45",    "pen": "0.25p,blue@40",      "label": "water"},
    "waterway": {"fill": None,            "pen": "0.65p,blue",         "label": "waterway"},
    "natural":  {"fill": "darkgreen@75",  "pen": "0.25p,darkgreen@50", "label": "natural"},
    "landuse":  {"fill": "lightgreen@70", "pen": "0.25p,darkgreen@40", "label": "landuse"},
    "aeroway":  {"fill": "orange@65",     "pen": "0.40p,orange",      "label": "aeroway"},
    "railway":  {"fill": None,            "pen": "0.80p,black",       "label": "railway"},
    "amenity":  {"fill": "yellow@60",     "pen": "0.25p,orange@50",   "label": "amenity"},
    "man_made": {"fill": "gray70@70",     "pen": "0.25p,gray40",      "label": "man_made"},
    "leisure":  {"fill": "lightcyan@65",  "pen": "0.25p,cyan@50",     "label": "leisure"},
    "barrier":  {"fill": None,            "pen": "0.60p,brown",       "label": "barrier"},
    "building": {"fill": "lightred@70",   "pen": "0.20p,black@35",    "label": "building"},
}

# Clip vector layers to AOI only for plotting
# True  = only plot features inside AOI polygon
# False = plot full geometry from input GPKG
CLIP_ROADS_TO_AOI_FOR_PLOT = True
CLIP_BUILDINGS_TO_AOI_FOR_PLOT = True
CLIP_OSM_EXTRA_TO_AOI_FOR_PLOT = True
# Plot buildings by height if height column is available. Otherwise plot all with same style.
PLOT_BUILDINGS_BY_HEIGHT = True
BUILDING_HEIGHT_COLUMN = "height_m"
# ============================================================
# 2. BASIC HELPERS
# ============================================================

def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def make_map_title(text: str) -> str:
    return f'WSen+t"{text}"'


def get_aoi_gdf() -> gpd.GeoDataFrame:
    if AOI_GPKG.exists():
        aoi = gpd.read_file(AOI_GPKG)
        if aoi.crs is None:
            aoi = aoi.set_crs("EPSG:4326")
        return aoi.to_crs("EPSG:4326")

    geom = Polygon(HOALAC_POLYGON)
    if not geom.is_valid:
        geom = geom.buffer(0)

    return gpd.GeoDataFrame(
        {"name": ["HoaLac_polygon_fallback"]},
        geometry=[geom],
        crs="EPSG:4326",
    )


def get_region_from_aoi(aoi: gpd.GeoDataFrame, padding: float = REGION_PADDING) -> list[float]:
    west, south, east, north = aoi.to_crs("EPSG:4326").total_bounds
    return [
        float(west - padding),
        float(east + padding),
        float(south - padding),
        float(north + padding),
    ]


def plot_aoi_boundary(fig: pygmt.Figure, aoi: gpd.GeoDataFrame, pen: str = POLYGON_PEN) -> None:
    aoi = aoi.to_crs("EPSG:4326")

    for geom in aoi.geometry:
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue

        for poly in polys:
            x, y = poly.exterior.xy
            fig.plot(x=list(x), y=list(y), pen=pen)


def start_map(region: list[float], title: str) -> pygmt.Figure:
    fig = pygmt.Figure()
    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            make_map_title(title),
            "xaf+lLongitude",
            "yaf+lLatitude",
        ],
    )
    return fig


def save_fig(fig: pygmt.Figure, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    print(f"[OK] Saved: {out_png}")


def read_gpkg_if_exists(path: Path) -> gpd.GeoDataFrame | None:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    return gdf.to_crs("EPSG:4326")

def clip_gdf_to_aoi_for_plot(
    gdf: gpd.GeoDataFrame | None,
    aoi: gpd.GeoDataFrame,
    enabled: bool = True,
    layer_name: str = "layer",
) -> gpd.GeoDataFrame | None:
    """
    Clip vector layer to AOI polygon only for plotting.
    Original GPKG files are not modified.
    """
    if gdf is None or gdf.empty:
        return gdf

    if not enabled:
        print(f"[INFO] Plot-time clipping OFF for {layer_name}")
        return gdf.to_crs("EPSG:4326")

    print(f"[INFO] Plot-time clipping ON for {layer_name}")

    gdf_wgs84 = gdf.to_crs("EPSG:4326").copy()
    aoi_wgs84 = aoi.to_crs("EPSG:4326").copy()

    n_before = len(gdf_wgs84)

    try:
        clipped = gpd.clip(gdf_wgs84, aoi_wgs84)
    except Exception as e:
        print(f"[WARN] Failed to clip {layer_name}: {e}")
        return gdf_wgs84

    clipped = clipped[
        clipped.geometry.notna() &
        (~clipped.geometry.is_empty)
    ].copy()

    print(f"[INFO] {layer_name} clipped features: {n_before} -> {len(clipped)}")

    return clipped

def safe_iter_geometries(gdf: gpd.GeoDataFrame):
    if gdf is None or gdf.empty:
        return

    gdf = gdf.to_crs("EPSG:4326")

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        yield geom


def plot_lines_from_gdf(fig: pygmt.Figure, gdf: gpd.GeoDataFrame, pen: str = ROAD_PEN) -> None:
    if gdf is None or gdf.empty:
        return

    for geom in safe_iter_geometries(gdf):
        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            coords = np.asarray(line.coords)
            if coords.shape[0] >= 2:
                fig.plot(x=coords[:, 0], y=coords[:, 1], pen=pen)


def plot_polygons_from_gdf(
    fig: pygmt.Figure,
    gdf: gpd.GeoDataFrame,
    fill: str | None = BUILDING_FILL,
    pen: str | None = BUILDING_PEN,
) -> None:
    if gdf is None or gdf.empty:
        return

    for geom in safe_iter_geometries(gdf):
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue

        for poly in polys:
            x, y = poly.exterior.xy
            fig.plot(x=list(x), y=list(y), fill=fill, pen=pen)


def plot_points_from_gdf(
    fig: pygmt.Figure,
    gdf: gpd.GeoDataFrame,
    style: str = EXTRA_POINT_STYLE,
    fill: str = EXTRA_POINT_FILL,
    pen: str = "0.1p,black",
) -> None:
    if gdf is None or gdf.empty:
        return

    point_gdf = gdf.to_crs("EPSG:4326").copy()
    point_gdf["geometry"] = point_gdf.geometry.representative_point()

    fig.plot(
        x=point_gdf.geometry.x,
        y=point_gdf.geometry.y,
        style=style,
        fill=fill,
        pen=pen,
    )


# ============================================================
# 3. ROAD CLASS HELPERS
# ============================================================

def parse_possible_list_value(value):
    """
    OSMnx may save list-like highway values as Python lists or strings that
    look like lists. This function extracts the first usable value.
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return value[0]

    if isinstance(value, str):
        value2 = value.strip()
        if value2.startswith("[") and value2.endswith("]"):
            try:
                parsed = ast.literal_eval(value2)
                if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                    return parsed[0]
            except Exception:
                pass
        return value2

    if pd.isna(value):
        return None

    return str(value)


def normalize_road_class(row) -> str:
    """
    Use simplified road_class from the download script when available.
    Otherwise use OSM highway.
    """
    for col in ["road_class", "highway_simple", "highway"]:
        if col in row.index:
            val = parse_possible_list_value(row[col])
            if val is not None:
                val = str(val).strip().lower()
                if val in ROAD_CLASS_STYLE:
                    return val

                # Map raw OSM *_link classes to their base class.
                if val.endswith("_link"):
                    base = val.replace("_link", "")
                    if base in ROAD_CLASS_STYLE:
                        return base

                # Map raw minor classes to our simplified buckets.
                if val in ["footway", "cycleway", "pedestrian", "path", "steps", "bridleway"]:
                    return val if val in ROAD_CLASS_STYLE else "non_motorized"

                return "other"

    return "other"


def plot_lines_by_road_class(fig: pygmt.Figure, roads_gdf: gpd.GeoDataFrame) -> list[str]:
    if roads_gdf is None or roads_gdf.empty:
        return []

    roads = roads_gdf.to_crs("EPSG:4326").copy()
    roads["road_class_plot"] = roads.apply(normalize_road_class, axis=1)

    used_classes = []

    for road_class in ROAD_PLOT_ORDER:
        sub = roads[roads["road_class_plot"] == road_class]
        if sub.empty:
            continue

        pen = ROAD_CLASS_STYLE.get(road_class, ROAD_CLASS_STYLE["other"])["pen"]

        for geom in safe_iter_geometries(sub):
            if geom.geom_type == "LineString":
                lines = [geom]
            elif geom.geom_type == "MultiLineString":
                lines = list(geom.geoms)
            else:
                continue

            for line in lines:
                coords = np.asarray(line.coords)
                if coords.shape[0] >= 2:
                    fig.plot(x=coords[:, 0], y=coords[:, 1], pen=pen)

        used_classes.append(road_class)

    return used_classes


def add_road_class_legend(fig: pygmt.Figure, used_classes: list[str]) -> None:
    if not used_classes:
        return

    dummy_x = [-1000, -999]
    dummy_y = [-1000, -1000]

    for road_class in used_classes:
        style = ROAD_CLASS_STYLE.get(road_class, ROAD_CLASS_STYLE["other"])
        fig.plot(
            x=dummy_x,
            y=dummy_y,
            pen=style["pen"],
            label=style["label"],
        )

    fig.legend(position=ROAD_LEGEND_POSITION, box=ROAD_LEGEND_BOX)


# ============================================================
# 4. RASTER HELPERS
# ============================================================

def robust_zrange_from_raster(raster_path: Path, lower: float = 2, upper: float = 98) -> tuple[float, float]:
    with rasterio.open(raster_path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        return 0.0, 1.0

    zmin = float(np.nanpercentile(vals, lower))
    zmax = float(np.nanpercentile(vals, upper))

    if np.isclose(zmin, zmax):
        zmin = float(np.nanmin(vals))
        zmax = float(np.nanmax(vals))

    if np.isclose(zmin, zmax):
        zmin -= 1.0
        zmax += 1.0

    return zmin, zmax


def reproject_raster_to_wgs84_if_needed(raster_path: Path) -> Path:
    """
    PyGMT map region/projection is lon/lat.
    If input raster is projected, reproject it to EPSG:4326 in TMP_DIR.
    """
    raster_path = Path(raster_path)

    with rasterio.open(raster_path) as src:
        src_crs = src.crs

        if src_crs is not None and src_crs.to_epsg() == 4326:
            return raster_path

        out_path = TMP_DIR / f"{raster_path.stem}_epsg4326.tif"

        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path

        dst_crs = "EPSG:4326"

        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
        )

        profile = src.profile.copy()
        profile.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "compress": "lzw",
            }
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            for band_id in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, band_id),
                    destination=rasterio.band(dst, band_id),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

    print(f"[OK] Reprojected raster to WGS84: {out_path}")
    return out_path


def clip_raster_to_aoi_wgs84(raster_path: Path, aoi: gpd.GeoDataFrame) -> Path:
    """
    Clip raster to AOI after ensuring raster is EPSG:4326.
    """
    raster_path = reproject_raster_to_wgs84_if_needed(raster_path)
    out_path = TMP_DIR / f"{raster_path.stem}_clip_aoi.tif"

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    aoi_wgs84 = aoi.to_crs("EPSG:4326")

    with rasterio.open(raster_path) as src:
        shapes = [mapping(geom) for geom in aoi_wgs84.geometry if geom is not None and not geom.is_empty]

        out_image, out_transform = rasterio.mask.mask(
            src,
            shapes,
            crop=True,
            nodata=src.nodata,
            filled=True,
        )

        profile = src.profile.copy()
        profile.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
            }
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_image)

    print(f"[OK] Clipped raster to AOI: {out_path}")
    return out_path


def plot_raster_map(
    raster_path: Path,
    aoi: gpd.GeoDataFrame,
    region: list[float],
    out_png: Path,
    title: str,
    cmap: str,
    label: str,
    continuous: bool = True,
    show_colorbar: bool = True,
    fixed_zrange: tuple[float, float] | None = None,
    overlay_roads: gpd.GeoDataFrame | None = None,
    overlay_buildings: gpd.GeoDataFrame | None = None,
) -> None:
    if not raster_path.exists():
        print(f"[WARN] Raster missing, skip: {raster_path}")
        return

    clipped = clip_raster_to_aoi_wgs84(raster_path, aoi)

    if fixed_zrange is None:
        zmin, zmax = robust_zrange_from_raster(clipped)
    else:
        zmin, zmax = fixed_zrange

    if not np.isfinite(zmin) or not np.isfinite(zmax) or np.isclose(zmin, zmax):
        zmin, zmax = 0.0, 1.0

    step = (zmax - zmin) / 100.0
    if not np.isfinite(step) or step <= 0:
        step = 1.0

    fig = start_map(region, title)

    pygmt.makecpt(
        cmap=cmap,
        series=[zmin, zmax, step],
        continuous=continuous,
    )

    fig.grdimage(
        grid=str(clipped),
        cmap=True,
        nan_transparent=True,
    )

    if OVERLAY_BUILDINGS_ON_RASTERS and overlay_buildings is not None:
        plot_polygons_from_gdf(fig, overlay_buildings, fill="lightred@80", pen="0.15p,black@60")

    if OVERLAY_ROADS_ON_RASTERS and overlay_roads is not None:
        plot_lines_from_gdf(fig, overlay_roads, pen="0.35p,black@30")

    if OVERLAY_AOI_ON_ALL_MAPS:
        plot_aoi_boundary(fig, aoi)

    if show_colorbar:
        fig.colorbar(
            frame=f'af+l"{label}"',
            position="JBC+w10c/0.35c+h+o0c/0.8c",
        )

    save_fig(fig, out_png)


# ============================================================
# 5. INDIVIDUAL MAPS
# ============================================================

def plot_overview_map(
    aoi: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame | None,
    buildings: gpd.GeoDataFrame | None,
    region: list[float],
    out_png: Path,
) -> None:
    print("\n[INFO] Creating overview map")

    fig = start_map(region, "Hoa Lac overview map: Buildings + Roads")

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

    plot_aoi_boundary(fig, aoi)

    # Legend
    fig.plot(x=[-1000], y=[-1000], style="s0.25c", fill=BUILDING_FILL, pen=BUILDING_PEN, label="Building polygon")
    fig.plot(x=[-1000, -999], y=[-1000, -1000], pen=ROAD_PEN, label="Road")
    fig.plot(x=[-1000, -999], y=[-1000, -1000], pen=POLYGON_PEN, label="Hoa Lac boundary")

    fig.legend(
        position=LEGEND_POSITION,
        box=f"+g{LEGEND_BOX_FILL}+p{LEGEND_BOX_PEN}",
    )

    save_fig(fig, out_png)


def plot_aoi_map(aoi: gpd.GeoDataFrame, region: list[float], out_png: Path) -> None:
    print("\n[INFO] Creating AOI map")

    fig = start_map(region, "Hoa Lac study-area AOI")

    # Fill AOI polygon lightly
    plot_polygons_from_gdf(fig, aoi, fill="purple@85", pen=POLYGON_PEN)
    plot_aoi_boundary(fig, aoi)

    fig.plot(x=[-1000], y=[-1000], style="s0.25c", fill="purple@85", pen=POLYGON_PEN, label="Study area")
    fig.legend(position=LEGEND_POSITION, box=f"+g{LEGEND_BOX_FILL}+p{LEGEND_BOX_PEN}")

    save_fig(fig, out_png)


def plot_roads_class_map(
    aoi: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame | None,
    region: list[float],
    out_png: Path,
) -> None:
    print("\n[INFO] Creating road-class map")

    if roads is None or roads.empty:
        print("[WARN] Roads are missing or empty, skip road-class map.")
        return

    fig = start_map(region, "OSM roads by road class")

    used_classes = plot_lines_by_road_class(fig, roads)
    plot_aoi_boundary(fig, aoi)
    add_road_class_legend(fig, used_classes)

    save_fig(fig, out_png)


def plot_road_summary_csv(csv_file: Path, out_png: Path) -> None:
    print("\n[INFO] Creating road summary chart")

    if not csv_file.exists():
        print(f"[WARN] Road summary CSV missing, skip: {csv_file}")
        return

    df = pd.read_csv(csv_file)

    if df.empty:
        print("[WARN] Road summary CSV is empty, skip.")
        return

    # Prefer total_length_km if available.
    if "total_length_km" not in df.columns:
        if "total_length_m" in df.columns:
            df["total_length_km"] = df["total_length_m"] / 1000.0
        else:
            print("[WARN] No length column in road summary CSV, skip chart.")
            return

    df = df.sort_values("total_length_km", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["road_class"].astype(str), df["total_length_km"])
    ax.set_xlabel("Total road length (km)")
    ax.set_ylabel("Road class")
    ax.set_title("Hoa Lac OSM road-class summary")
    ax.grid(axis="x", alpha=0.3)

    for i, v in enumerate(df["total_length_km"]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)

    print(f"[OK] Saved: {out_png}")


def select_extra_tag(row) -> str:
    """
    Pick one tag to represent an OSM extra feature.
    """
    for tag in EXTRA_TAG_PRIORITY:
        if tag in row.index:
            val = row[tag]
            if val is not None and not pd.isna(val):
                return tag
    return "other"


def plot_osm_extra_features_map(
    aoi: gpd.GeoDataFrame,
    extra: gpd.GeoDataFrame | None,
    region: list[float],
    out_png: Path,
) -> None:
    print("\n[INFO] Creating OSM extra-features map")

    if extra is None or extra.empty:
        print("[WARN] OSM extra features missing or empty, skip.")
        return

    extra = extra.to_crs("EPSG:4326").copy()
    extra["plot_tag"] = extra.apply(select_extra_tag, axis=1)

    fig = start_map(region, "OSM extra features")

    used_tags = []

    for tag in EXTRA_TAG_PRIORITY:
        sub = extra[extra["plot_tag"] == tag]
        if sub.empty:
            continue

        style = EXTRA_TAG_STYLE.get(tag, {})
        fill = style.get("fill", EXTRA_POLYGON_FILL)
        pen = style.get("pen", EXTRA_LINE_PEN)

        poly_gdf = sub[sub.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        line_gdf = sub[sub.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        point_gdf = sub[sub.geometry.geom_type.isin(["Point", "MultiPoint"])]

        if not poly_gdf.empty:
            plot_polygons_from_gdf(fig, poly_gdf, fill=fill, pen=pen)

        if not line_gdf.empty:
            plot_lines_from_gdf(fig, line_gdf, pen=pen)

        if not point_gdf.empty:
            plot_points_from_gdf(fig, point_gdf, fill=EXTRA_POINT_FILL, pen="0.1p,black")

        used_tags.append(tag)

    plot_aoi_boundary(fig, aoi)

    # Legend using dummy objects
    for tag in used_tags:
        style = EXTRA_TAG_STYLE.get(tag, {})
        fill = style.get("fill", EXTRA_POLYGON_FILL)
        pen = style.get("pen", EXTRA_LINE_PEN)
        label = style.get("label", tag)

        if fill is None:
            fig.plot(x=[-1000, -999], y=[-1000, -1000], pen=pen, label=label)
        else:
            fig.plot(x=[-1000], y=[-1000], style="s0.23c", fill=fill, pen=pen, label=label)

    fig.legend(position=LEGEND_POSITION, box=f"+g{LEGEND_BOX_FILL}+p{LEGEND_BOX_PEN}")

    save_fig(fig, out_png)


def plot_building_map(
    aoi: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None,
    region: list[float],
    out_png: Path,
) -> None:
    print("\n[INFO] Creating OpenBuildingMap buildings map")

    if buildings is None or buildings.empty:
        print("[WARN] OBM building file missing or empty, skip.")
        return

    buildings = buildings.to_crs("EPSG:4326").copy()

    fig = start_map(region, "OpenBuildingMap buildings")

    if PLOT_BUILDINGS_BY_HEIGHT and BUILDING_HEIGHT_COLUMN in buildings.columns:
        buildings[BUILDING_HEIGHT_COLUMN] = pd.to_numeric(
            buildings[BUILDING_HEIGHT_COLUMN],
            errors="coerce",
        )

        valid = buildings.dropna(subset=[BUILDING_HEIGHT_COLUMN]).copy()

        if valid.empty:
            print(f"[WARN] No valid numeric values in {BUILDING_HEIGHT_COLUMN}. Plot normal buildings.")
            plot_polygons_from_gdf(fig, buildings, fill=BUILDING_FILL, pen=BUILDING_PEN)

        else:
            zmin = float(valid[BUILDING_HEIGHT_COLUMN].quantile(0.02))
            zmax = float(valid[BUILDING_HEIGHT_COLUMN].quantile(0.98))

            if np.isclose(zmin, zmax):
                zmin = float(valid[BUILDING_HEIGHT_COLUMN].min())
                zmax = float(valid[BUILDING_HEIGHT_COLUMN].max())

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
                value = row[BUILDING_HEIGHT_COLUMN]

                if geom is None or geom.is_empty:
                    continue

                if geom.geom_type == "Polygon":
                    polys = [geom]
                elif geom.geom_type == "MultiPolygon":
                    polys = list(geom.geoms)
                else:
                    continue

                for poly in polys:
                    x, y = poly.exterior.xy
                    fig.plot(
                        x=list(x),
                        y=list(y),
                        fill=value,
                        cmap=True,
                        pen=BUILDING_PEN,
                    )

            fig.colorbar(
                frame=f'af+l"Building height (m)"',
                position="JBC+w10c/0.35c+h+o0c/0.8c",
            )

    else:
        plot_polygons_from_gdf(fig, buildings, fill=BUILDING_FILL, pen=BUILDING_PEN)

    plot_aoi_boundary(fig, aoi)

    fig.plot(
        x=[-1000, -999],
        y=[-1000, -1000],
        pen=POLYGON_PEN,
        label="Hoa Lac boundary",
    )

    fig.legend(
        position=LEGEND_POSITION,
        box=f"+g{LEGEND_BOX_FILL}+p{LEGEND_BOX_PEN}",
    )

    save_fig(fig, out_png)


# ============================================================
# 6. MAIN
# ============================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_dirs()

    print("\n========== PLOT HOA LAC STUDY AREA ==========")
    print(f"[INFO] Input data dir: {DATA_DIR}")
    print(f"[INFO] Output fig dir: {FIG_DIR}")

    aoi = get_aoi_gdf()
    region = get_region_from_aoi(aoi, padding=REGION_PADDING)

    roads = read_gpkg_if_exists(ROADS_GPKG)
    extra = read_gpkg_if_exists(OSM_EXTRA_GPKG)
    buildings = read_gpkg_if_exists(OBM_BUILDINGS_GPKG)

    # Optional plot-time clipping to AOI
    roads = clip_gdf_to_aoi_for_plot(
        roads,
        aoi,
        enabled=CLIP_ROADS_TO_AOI_FOR_PLOT,
        layer_name="OSM roads",
    )

    extra = clip_gdf_to_aoi_for_plot(
        extra,
        aoi,
        enabled=CLIP_OSM_EXTRA_TO_AOI_FOR_PLOT,
        layer_name="OSM extra features",
    )

    buildings = clip_gdf_to_aoi_for_plot(
        buildings,
        aoi,
        enabled=CLIP_BUILDINGS_TO_AOI_FOR_PLOT,
        layer_name="OBM buildings",
    )

    print(f"[INFO] Plot region: {region}")

    # Keep overview first, then keep the order of the downloaded output files.
    plot_jobs = [
        ("00_overview_map.png", plot_overview_map, {}),
        ("01_study_area_aoi.png", plot_aoi_map, {}),
        ("02_osm_roads_edges_by_class.png", plot_roads_class_map, {}),
        ("03_osm_road_class_summary.png", plot_road_summary_csv, {}),
        ("04_osm_extra_features.png", plot_osm_extra_features_map, {}),
        ("05_opentopography_SRTMGL1_dem_wgs84.png", "raster", {
            "raster_path": DEM_WGS84_TIF,
            "title": "OpenTopography SRTMGL1 DEM WGS84",
            "cmap": "geo",
            "label": "Elevation (m)",
            "continuous": True,
            "show_colorbar": True,
        }),
        ("06_opentopography_SRTMGL1_dem_utm.png", "raster", {
            "raster_path": DEM_UTM_TIF,
            "title": "OpenTopography SRTMGL1 DEM UTM",
            "cmap": "geo",
            "label": "Elevation (m)",
            "continuous": True,
            "show_colorbar": True,
        }),
        ("07_slope_degree.png", "raster", {
            "raster_path": SLOPE_DEGREE_TIF,
            "title": "Slope from OpenTopography DEM",
            "cmap": "turbo",
            "label": "Slope (degree)",
            "continuous": True,
            "show_colorbar": True,
            "fixed_zrange": None,
        }),
        ("08_hillshade.png", "raster", {
            "raster_path": HILLSHADE_TIF,
            "title": "Hillshade from OpenTopography DEM",
            "cmap": "gray",
            "label": "Hillshade",
            "continuous": True,
            "show_colorbar": True,
            "fixed_zrange": (0.0, 255.0),
        }),
        ("09_obm_buildings_hoalac_clipped.png", plot_building_map, {}),
    ]

    for out_name, job, kwargs in plot_jobs:
        out_png = FIG_DIR / out_name

        if job == plot_overview_map:
            job(aoi=aoi, roads=roads, buildings=buildings, region=region, out_png=out_png)

        elif job == plot_aoi_map:
            job(aoi=aoi, region=region, out_png=out_png)

        elif job == plot_roads_class_map:
            job(aoi=aoi, roads=roads, region=region, out_png=out_png)

        elif job == plot_road_summary_csv:
            job(csv_file=ROAD_SUMMARY_CSV, out_png=out_png)

        elif job == plot_osm_extra_features_map:
            job(aoi=aoi, extra=extra, region=region, out_png=out_png)

        elif job == plot_building_map:
            job(aoi=aoi, buildings=buildings, region=region, out_png=out_png)

        elif job == "raster":
            plot_raster_map(
                aoi=aoi,
                region=region,
                out_png=out_png,
                overlay_roads=roads,
                overlay_buildings=buildings,
                **kwargs,
            )

        else:
            raise ValueError(f"Unknown plot job: {job}")

    print("\n========== DONE ==========")
    print(f"Figures saved in: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
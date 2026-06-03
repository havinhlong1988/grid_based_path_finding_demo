#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert manually exported Google Earth Pro road/path KML/KMZ
to GIS + XYZ products for Hoa Lac Hi-Tech Park comparison.

Important:
    This script does NOT scrape Google Maps or Google Earth road data.
    It only converts user-created/exported KML/KMZ paths from Google Earth Pro.

Workflow:
    1. Open Google Earth Pro.
    2. Manually draw road/path lines using Add Path.
    3. Put all paths in one folder, e.g. "google_earth_roads_hoalac".
    4. Right-click folder -> Save Place As -> KML or KMZ.
    5. Set INPUT_KML_OR_KMZ below.
    6. Run this script.

Outputs:
    output_hoalac_googleearth_roads/
    ├── hoalac_polygon.gpkg
    ├── googleearth_roads_raw.gpkg
    ├── googleearth_roads_hoalac_clipped.gpkg
    ├── googleearth_roads_hoalac_clipped.geojson
    ├── googleearth_roads_vertices_hoalac.xyz
    └── googleearth_roads_summary.csv

XYZ format:
    lon lat road_code segment_id

Since Google Earth Pro manually drawn paths do not have official road class,
all paths are assigned:
    road_class = "googleearth_path"
    road_code  = 90
"""

from pathlib import Path
import zipfile
import tempfile
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LineString, MultiLineString


# ============================================================
# USER INPUT PARAMETERS
# ============================================================

# Put your Google Earth Pro exported KML/KMZ here
INPUT_KML_OR_KMZ = Path("googleearth_roads_hoalac.kml")

OUTDIR = Path("output_hoalac_googleearth_roads")

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

DEFAULT_ROAD_CLASS = "googleearth_path"
DEFAULT_ROAD_CODE = 90


# ============================================================
# OUTPUT FILES
# ============================================================

POLYGON_GPKG = OUTDIR / "hoalac_polygon.gpkg"
RAW_GPKG = OUTDIR / "googleearth_roads_raw.gpkg"
CLIPPED_GPKG = OUTDIR / "googleearth_roads_hoalac_clipped.gpkg"
CLIPPED_GEOJSON = OUTDIR / "googleearth_roads_hoalac_clipped.geojson"
VERTICES_XYZ = OUTDIR / "googleearth_roads_vertices_hoalac.xyz"
SUMMARY_CSV = OUTDIR / "googleearth_roads_summary.csv"


# ============================================================
# HELPERS
# ============================================================

def make_hoalac_polygon_gdf():
    poly = Polygon(HOALAC_POLYGON)
    if not poly.is_valid:
        poly = poly.buffer(0)

    return gpd.GeoDataFrame(
        {"name": ["Hoa_Lac_HiTech_Park_approx"]},
        geometry=[poly],
        crs="EPSG:4326",
    )


def extract_kmz_to_kml(kmz_file):
    """
    Extract doc.kml from KMZ.
    """
    kmz_file = Path(kmz_file)

    tmpdir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmpdir.name)

    with zipfile.ZipFile(kmz_file, "r") as zf:
        kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]

        if not kml_names:
            raise RuntimeError(f"No KML found inside KMZ: {kmz_file}")

        # Usually doc.kml is the main file
        main_kml = "doc.kml" if "doc.kml" in kml_names else kml_names[0]

        zf.extract(main_kml, tmp_path)

    return tmp_path / main_kml, tmpdir


def read_kml_or_kmz(input_file):
    """
    Read KML/KMZ using GeoPandas/Fiona/Pyogrio.

    Returns GeoDataFrame in EPSG:4326.
    """
    input_file = Path(input_file)

    if not input_file.exists():
        raise FileNotFoundError(
            f"\nInput KML/KMZ not found:\n{input_file}\n\n"
            "Please export paths from Google Earth Pro and set INPUT_KML_OR_KMZ."
        )

    tmpdir_obj = None

    if input_file.suffix.lower() == ".kmz":
        kml_file, tmpdir_obj = extract_kmz_to_kml(input_file)
    else:
        kml_file = input_file

    print(f"[INFO] Reading KML: {kml_file}")

    # GeoPandas may read KML directly if GDAL/Fiona has KML driver enabled.
    try:
        gdf = gpd.read_file(kml_file, driver="KML")
    except Exception:
        # Fallback: let engine auto-detect
        gdf = gpd.read_file(kml_file)

    if gdf.empty:
        raise RuntimeError(f"No features found in KML/KMZ: {input_file}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    if tmpdir_obj is not None:
        # Keep temp alive until gdf loaded; gdf is already in memory now.
        tmpdir_obj.cleanup()

    return gdf


def keep_only_lines(gdf):
    """
    Keep LineString/MultiLineString features from KML.
    """
    lines = gdf[
        gdf.geometry.notna()
        & (~gdf.geometry.is_empty)
        & gdf.geometry.type.isin(["LineString", "MultiLineString"])
    ].copy()

    if lines.empty:
        raise RuntimeError(
            "No LineString/MultiLineString found in KML.\n"
            "In Google Earth Pro, use 'Add Path', not only polygon/placemark."
        )

    lines = lines.reset_index(drop=True)

    if "Name" in lines.columns:
        lines["name"] = lines["Name"].astype(str)
    elif "name" not in lines.columns:
        lines["name"] = [f"googleearth_path_{i:04d}" for i in range(len(lines))]

    lines["road_class"] = DEFAULT_ROAD_CLASS
    lines["road_code"] = DEFAULT_ROAD_CODE
    lines["source"] = "google_earth_pro_manual_kml"

    return lines


def clip_roads_to_polygon(roads_gdf, polygon_gdf):
    roads = roads_gdf.to_crs("EPSG:4326").copy()
    poly = polygon_gdf.to_crs("EPSG:4326").copy()

    clipped = gpd.clip(roads, poly)

    if clipped.empty:
        raise RuntimeError(
            "No Google Earth paths intersect Hoa Lac polygon after clipping.\n"
            "Check your KML path location and the HOALAC_POLYGON coordinates."
        )

    return clipped.reset_index(drop=True)


def save_vertices_xyz(roads_gdf, out_xyz):
    """
    Save road/path vertices:
        lon lat road_code segment_id
    """
    out_xyz = Path(out_xyz)

    roads = roads_gdf.to_crs("EPSG:4326").copy().reset_index(drop=True)

    records = []
    segment_id = 0

    for _, row in roads.iterrows():
        geom = row.geometry
        road_code = int(row.get("road_code", DEFAULT_ROAD_CODE))

        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            coords = list(line.coords)

            if len(coords) < 2:
                continue

            for x, y in coords:
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

    print(f"[OK] Saved XYZ: {out_xyz}")
    print("     Format: lon lat road_code segment_id")


def save_summary(roads_gdf, out_csv):
    rows = []

    rows.append({
        "metric": "n_features",
        "value": len(roads_gdf),
    })

    roads_m = roads_gdf.to_crs(roads_gdf.estimate_utm_crs())
    lengths_m = roads_m.geometry.length

    rows.append({
        "metric": "total_length_m",
        "value": float(lengths_m.sum()),
    })

    rows.append({
        "metric": "total_length_km",
        "value": float(lengths_m.sum() / 1000.0),
    })

    if "name" in roads_gdf.columns:
        rows.append({
            "metric": "n_named_paths",
            "value": int(roads_gdf["name"].notna().sum()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"[OK] Saved summary: {out_csv}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("\n========== CONVERT GOOGLE EARTH PRO KML ROADS ==========")

    polygon_gdf = make_hoalac_polygon_gdf()
    polygon_gdf.to_file(POLYGON_GPKG, driver="GPKG")
    print(f"[OK] Saved polygon: {POLYGON_GPKG}")

    raw_gdf = read_kml_or_kmz(INPUT_KML_OR_KMZ)
    print(f"[INFO] Raw KML features: {len(raw_gdf)}")
    print(f"[INFO] Raw geometry types:")
    print(raw_gdf.geometry.type.value_counts())

    roads_raw = keep_only_lines(raw_gdf)
    roads_raw.to_file(RAW_GPKG, driver="GPKG")
    print(f"[OK] Saved raw paths: {RAW_GPKG}")

    roads_clip = clip_roads_to_polygon(roads_raw, polygon_gdf)

    roads_clip.to_file(CLIPPED_GPKG, driver="GPKG")
    roads_clip.to_file(CLIPPED_GEOJSON, driver="GeoJSON")

    print(f"[OK] Saved clipped GPKG: {CLIPPED_GPKG}")
    print(f"[OK] Saved clipped GeoJSON: {CLIPPED_GEOJSON}")

    save_vertices_xyz(roads_clip, VERTICES_XYZ)
    save_summary(roads_clip, SUMMARY_CSV)

    print("\n========== DONE ==========")
    print(f"Output folder: {OUTDIR.resolve()}")
    print("")
    print("Use in PyGMT plot script:")
    print(f"  DATA_DIR = Path('{OUTDIR}')")
    print(f"  ROADS_GPKG = DATA_DIR / '{CLIPPED_GPKG.name}'")
    print(f"  road XYZ = {VERTICES_XYZ}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
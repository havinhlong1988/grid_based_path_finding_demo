#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create nested-base model for Scenario 1 no velocity.

Run:
    python 06_create_nested_base_model_senario1_no_velocity.py

Output model:
    output/02_senario1_no_velocity/raw.xyz

Output columns:
    longitude latitude elevation_m slowness_s_per_m

Logic:
    inside Hoa Lac polygon  -> slowness = 0.02 s/m
    outside Hoa Lac polygon -> slowness = 1e5 s/m

Figures:
    figures/02_senario1_no_velocity/model_2d_z0_categorical.png
    figures/02_senario1_no_velocity/model_3d_nodes_perspective.png
"""

from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from pyproj import Transformer

import pygmt


# ============================================================
# Main editable parameters
# ============================================================

PROJECT_DIR = Path(".").resolve()

INPUT_DIR = PROJECT_DIR / "input" / "02_data_senario1_no_velocity"
OUTPUT_XYZ = PROJECT_DIR / "output" / "02_senario1_no_velocity" / "raw.xyz"
OUTPUT_VTK = PROJECT_DIR / "output" / "02_senario1_no_velocity" / "raw.vtk"
OUTPUT_NODES_VTK = PROJECT_DIR / "output" / "02_senario1_no_velocity" / "model_nodes.vtk"
OUTPUT_POLYGON_VTK = PROJECT_DIR / "output" / "02_senario1_no_velocity" / "aoi_polygon_cage.vtk"
FIG_DIR = PROJECT_DIR / "figures" / "02_senario1_no_velocity"

# ------------------------------------------------------------
# Area / title / plot region setting
# ------------------------------------------------------------

# If True, use HOALAC_POLYGON directly and skip AOI_GPKG / KML / XYZ search.
USE_HOALAC_POLYGON = True

USE_PLACE_NAME = True
PLACE_NAME = "Hoa Lac Hi-Tech Park, Hanoi, Vietnam"

# Optional AOI GPKG.
# Used only if USE_HOALAC_POLYGON = False.
AOI_GPKG = INPUT_DIR / "aoi" / "HoaLac_AOI.gpkg"

# Optional explicit polygon file.
# Used only if USE_HOALAC_POLYGON = False.
POLYGON_FILE = None
# Example:
# POLYGON_FILE = INPUT_DIR / "kml_plan" / "HoaLac_boundary.xyz"

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

# ------------------------------------------------------------
# Model grid setting
# ------------------------------------------------------------

DX = 50.0
DY = 50.0
DZ = 50.0

ZMIN = 0.0
ZMAX = 3000.0

INSIDE_SLOWNESS = 2e-2
OUTSIDE_SLOWNESS = 1e5

# Hoa Lac / Hanoi: UTM zone 48N.
EPSG_UTM = 32648

# "min" -> local x/y from polygon minimum bound.
# "utm" -> full UTM x/y.
ORIGIN_MODE = "min"

# ------------------------------------------------------------
# Figure / surface setting
# ------------------------------------------------------------

PROJECTION = "M15c"
REGION_PADDING = 0.003
DPI = 300

FIG_3D_PROJECTION = "X15c/13c"
FIG_3D_ZSIZE = "7c"
FIG_3D_PERSPECTIVE = [135, 30]

CATEGORY_TRANSPARENCY = 50

DOT_SIZE_2D = "c0.035c"
DOT_SIZE_3D = "c0.035c"

POLYGON_RGB = "purple"
POLYGON_PEN_2D = f"1.8p,{POLYGON_RGB}"
POLYGON_PEN_3D = f"1.4p,{POLYGON_RGB}"
VERTEX_CONNECT_PEN_3D = f"0.8p,{POLYGON_RGB},-"

# Surface spacing for 2D categorical fill, in degree.
# If None, it will be estimated from DX/DY.
SURFACE_SPACING_DEG = None

DO_PLOT = True


# ============================================================
# Coordinate helpers
# ============================================================

def looks_like_lonlat(x, y):
    return (-180 <= x <= 180) and (-90 <= y <= 90)


def lonlat_to_utm(lon, lat, epsg=EPSG_UTM):
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def utm_to_lonlat(x, y, epsg=EPSG_UTM):
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return np.asarray(lon), np.asarray(lat)


def close_polygon_if_needed(coords):
    if len(coords) < 3:
        raise ValueError("Polygon needs at least 3 points.")

    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]

    return coords


def polygon_from_lonlat(lonlat_points):
    lon = np.array([p[0] for p in lonlat_points])
    lat = np.array([p[1] for p in lonlat_points])

    x, y = lonlat_to_utm(lon, lat)

    coords = list(zip(x.tolist(), y.tolist()))
    coords = close_polygon_if_needed(coords)

    poly = Polygon(coords)

    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty:
        raise ValueError("HOALAC_POLYGON is invalid.")

    return poly


def polygon_to_lonlat(poly):
    coords = np.asarray(poly.exterior.coords)
    lon, lat = utm_to_lonlat(coords[:, 0], coords[:, 1])
    return lon, lat


# ============================================================
# Read polygon from GPKG
# ============================================================

def read_polygon_from_gpkg(path):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is not installed. Install it or use HOALAC_POLYGON/KML/XYZ."
        ) from exc

    path = Path(path)
    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"GPKG is empty: {path}")

    gdf = gdf.to_crs(epsg=EPSG_UTM)
    geom = gdf.geometry.unary_union

    if geom.geom_type == "MultiPolygon":
        geom = max(list(geom.geoms), key=lambda g: g.area)

    if geom.geom_type != "Polygon":
        raise ValueError(f"GPKG geometry is not polygon: {geom.geom_type}")

    if not geom.is_valid:
        geom = geom.buffer(0)

    if geom.is_empty:
        raise ValueError(f"Invalid GPKG geometry: {path}")

    return geom


# ============================================================
# Read polygon from XYZ
# ============================================================

def read_polygon_from_xyz(path):
    path = Path(path)
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.replace(",", " ").split()

            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass

            if len(nums) < 2:
                continue

            label = parts[0] if len(parts) > len(nums) else ""
            x = nums[0]
            y = nums[1]

            rows.append((label, x, y))

    if len(rows) < 3:
        raise ValueError(f"Not enough polygon points in {path}")

    positive_keys = (
        "ring", "perimeter", "boundary", "polygon",
        "park", "area", "hoalac", "hoa_lac"
    )
    negative_keys = ("center", "centroid", "control")

    preferred = []
    fallback = []

    for label, x, y in rows:
        low = label.lower()

        if any(k in low for k in negative_keys):
            continue

        fallback.append((x, y))

        if any(k in low for k in positive_keys):
            preferred.append((x, y))

    coords_raw = preferred if len(preferred) >= 3 else fallback

    if len(coords_raw) < 3:
        raise ValueError(f"Cannot build polygon from {path}")

    coords_unique = []

    for xy in coords_raw:
        if not coords_unique or xy != coords_unique[-1]:
            coords_unique.append(xy)

    x0, y0 = coords_unique[0]

    if looks_like_lonlat(x0, y0):
        lon = np.array([p[0] for p in coords_unique])
        lat = np.array([p[1] for p in coords_unique])
        xx, yy = lonlat_to_utm(lon, lat)
        coords_unique = list(zip(xx.tolist(), yy.tolist()))

    coords_unique = close_polygon_if_needed(coords_unique)

    poly = Polygon(coords_unique)

    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty:
        raise ValueError(f"Invalid polygon from {path}")

    return poly


# ============================================================
# Read polygon from KML / KMZ
# ============================================================

def parse_kml_coordinates(kml_text):
    root = ET.fromstring(kml_text)

    coord_blocks = []

    for elem in root.iter():
        if elem.tag.endswith("coordinates") and elem.text:
            text = elem.text.strip()
            if text:
                coord_blocks.append(text)

    polygons = []

    for block in coord_blocks:
        pts = []

        for item in block.replace("\n", " ").replace("\t", " ").split():
            vals = item.split(",")

            if len(vals) < 2:
                continue

            try:
                lon = float(vals[0])
                lat = float(vals[1])
            except ValueError:
                continue

            pts.append((lon, lat))

        if len(pts) >= 3:
            lon = np.array([p[0] for p in pts])
            lat = np.array([p[1] for p in pts])
            xx, yy = lonlat_to_utm(lon, lat)

            coords = list(zip(xx.tolist(), yy.tolist()))
            coords = close_polygon_if_needed(coords)

            poly = Polygon(coords)

            if not poly.is_valid:
                poly = poly.buffer(0)

            if not poly.is_empty:
                polygons.append(poly)

    if not polygons:
        raise ValueError("No valid polygon found in KML.")

    polygons = sorted(polygons, key=lambda p: p.area, reverse=True)
    return polygons[0]


def read_polygon_from_kml(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    return parse_kml_coordinates(text)


def read_polygon_from_kmz(path):
    path = Path(path)

    with zipfile.ZipFile(path, "r") as z:
        kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]

        if not kml_names:
            raise ValueError(f"No KML file found inside {path}")

        kml_name = "doc.kml" if "doc.kml" in kml_names else kml_names[0]
        text = z.read(kml_name).decode("utf-8", errors="ignore")

    return parse_kml_coordinates(text)


def read_polygon(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".gpkg":
        return read_polygon_from_gpkg(path)

    if suffix == ".xyz":
        return read_polygon_from_xyz(path)

    if suffix == ".kml":
        return read_polygon_from_kml(path)

    if suffix == ".kmz":
        return read_polygon_from_kmz(path)

    raise ValueError(f"Unsupported polygon file type: {path}")


# ============================================================
# Polygon loading priority
# ============================================================

def auto_find_polygon(input_dir):
    input_dir = Path(input_dir)

    search_dirs = [
        input_dir / "kml_plan",
        PROJECT_DIR / "kml" / "plan",
        PROJECT_DIR / "output" / "01_HoaLac_studies_area" / "kml_plan",
    ]

    candidates = []

    for d in search_dirs:
        if not d.exists():
            continue

        for ext in ("*.xyz", "*.kml", "*.kmz"):
            candidates.extend(d.glob(ext))

    if not candidates:
        raise FileNotFoundError("No KML/XYZ/KMZ polygon file found automatically.")

    keywords = (
        "hoalac", "hoa_lac", "park", "boundary", "perimeter",
        "polygon", "area", "ring"
    )

    scored = []

    for f in candidates:
        name = f.name.lower()
        score = sum(k in name for k in keywords)

        if "centroid" in name or "center" in name:
            score -= 3

        scored.append((score, f))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)

    errors = []

    for score, f in scored:
        try:
            poly = read_polygon(f)
            if poly.area > 0:
                print(f"[OK] Auto polygon file: {f}")
                return poly, f
        except Exception as e:
            errors.append((f, str(e)))

    msg = "Found polygon-like files, but failed to read a valid polygon:\n"
    for f, e in errors:
        msg += f"  - {f}: {e}\n"

    raise RuntimeError(msg)


def load_area_polygon():
    """
    Loading priority:
        1. HOALAC_POLYGON if USE_HOALAC_POLYGON = True
        2. POLYGON_FILE
        3. AOI_GPKG
        4. auto search in kml_plan
        5. fallback HOALAC_POLYGON
    """
    if USE_HOALAC_POLYGON:
        print("[OK] Use HOALAC_POLYGON from script header.")
        poly = polygon_from_lonlat(HOALAC_POLYGON)
        return poly, "HOALAC_POLYGON"

    if POLYGON_FILE is not None:
        poly_file = Path(POLYGON_FILE)
        poly = read_polygon(poly_file)
        print(f"[OK] Polygon file from POLYGON_FILE: {poly_file}")
        return poly, poly_file

    aoi_gpkg = Path(AOI_GPKG)

    if aoi_gpkg.exists():
        try:
            poly = read_polygon_from_gpkg(aoi_gpkg)
            print(f"[OK] AOI GPKG polygon: {aoi_gpkg}")
            return poly, aoi_gpkg
        except Exception as e:
            print(f"[WARN] Failed to read AOI GPKG: {aoi_gpkg}")
            print(f"       Reason: {e}")

    try:
        poly, poly_file = auto_find_polygon(INPUT_DIR)
        return poly, poly_file
    except Exception as e:
        print("[WARN] Failed to find polygon from KML/XYZ/KMZ.")
        print(f"       Reason: {e}")

    print("[OK] Use fallback HOALAC_POLYGON from script header.")
    poly = polygon_from_lonlat(HOALAC_POLYGON)
    return poly, "HOALAC_POLYGON fallback"


# ============================================================
# Build model
# ============================================================

def make_grid_values(
    poly,
    dx,
    dy,
    dz,
    zmin,
    zmax,
    inside_slow,
    outside_slow,
    origin_mode,
):
    minx, miny, maxx, maxy = poly.bounds

    xs_utm = np.arange(minx, maxx + dx * 0.5, dx)
    ys_utm = np.arange(miny, maxy + dy * 0.5, dy)
    zs = np.arange(zmin, zmax + dz * 0.5, dz)

    prepared_poly = prep(poly)

    if origin_mode == "min":
        x0 = minx
        y0 = miny
    elif origin_mode == "utm":
        x0 = 0.0
        y0 = 0.0
    else:
        raise ValueError("ORIGIN_MODE must be 'min' or 'utm'")

    nx = len(xs_utm)
    ny = len(ys_utm)
    nz = len(zs)

    print("\nGrid size:")
    print(f"  nx = {nx}")
    print(f"  ny = {ny}")
    print(f"  nz = {nz}")
    print(f"  total points = {nx * ny * nz:,}")

    slow2d = np.zeros((ny, nx), dtype=float)
    mask2d = np.zeros((ny, nx), dtype=float)

    for iy, y in enumerate(ys_utm):
        for ix, x in enumerate(xs_utm):
            pt = Point(x, y)

            if prepared_poly.covers(pt):
                slow2d[iy, ix] = inside_slow
                mask2d[iy, ix] = 1.0
            else:
                slow2d[iy, ix] = outside_slow
                mask2d[iy, ix] = 0.0

    xs = xs_utm - x0
    ys = ys_utm - y0

    return xs, ys, xs_utm, ys_utm, zs, slow2d, mask2d, (x0, y0)


def save_raw_xyz(output_file, xs_utm, ys_utm, zs, slow2d):
    """
    Save raw XYZ model as geographic coordinates.

    Output columns:
        longitude latitude elevation_m slowness_s_per_m

    Notes:
        - longitude/latitude are converted from UTM model nodes.
        - elevation_m follows zs.
        - slowness_s_per_m follows the 2D slowness mask repeated at all elevations.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n = 0

    # Convert 2D UTM grid nodes to lon/lat once.
    xx_utm, yy_utm = np.meshgrid(xs_utm, ys_utm)
    lon2d, lat2d = utm_to_lonlat(xx_utm, yy_utm)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# longitude latitude elevation_m slowness_s_per_m\n")

        for z in zs:
            for iy in range(len(ys_utm)):
                for ix in range(len(xs_utm)):
                    lon = lon2d[iy, ix]
                    lat = lat2d[iy, ix]
                    s = slow2d[iy, ix]
                    f.write(f"{lon:.8f} {lat:.8f} {z:.3f} {s:.8g}\n")
                    n += 1

    return n

def save_vtk_rectilinear_grid(output_file, xs, ys, zs, slow2d, mask2d):
    """
    Save model to legacy ASCII VTK RECTILINEAR_GRID format.

    Coordinates:
        x = xs, local model x in meters
        y = ys, local model y in meters
        z = zs, elevation in meters

    Point data:
        slowness_s_per_m
        flyable_mask
        category

    Category:
        0 = flyable
        1 = no-fly
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    nx = len(xs)
    ny = len(ys)
    nz = len(zs)
    npoints = nx * ny * nz

    # Category:
    # 0 = flyable
    # 1 = no-fly
    category2d = np.where(mask2d == 1, 0, 1).astype(int)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Hoa Lac Scenario 1 no velocity slowness model\n")
        f.write("ASCII\n")
        f.write("DATASET RECTILINEAR_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")

        f.write(f"X_COORDINATES {nx} float\n")
        for x in xs:
            f.write(f"{x:.6f}\n")

        f.write(f"Y_COORDINATES {ny} float\n")
        for y in ys:
            f.write(f"{y:.6f}\n")

        f.write(f"Z_COORDINATES {nz} float\n")
        for z in zs:
            f.write(f"{z:.6f}\n")

        f.write(f"POINT_DATA {npoints}\n")

        # ----------------------------------------------------
        # Slowness field
        # VTK point order: x fastest, then y, then z.
        # This matches the raw.xyz loop order: z -> y -> x.
        # ----------------------------------------------------
        f.write("SCALARS slowness_s_per_m float 1\n")
        f.write("LOOKUP_TABLE default\n")

        for _z in zs:
            for iy in range(ny):
                for ix in range(nx):
                    f.write(f"{slow2d[iy, ix]:.8e}\n")

        # ----------------------------------------------------
        # Flyable mask
        # 1 = flyable, 0 = no-fly
        # ----------------------------------------------------
        f.write("SCALARS flyable_mask int 1\n")
        f.write("LOOKUP_TABLE default\n")

        for _z in zs:
            for iy in range(ny):
                for ix in range(nx):
                    f.write(f"{int(mask2d[iy, ix])}\n")

        # ----------------------------------------------------
        # Category
        # 0 = flyable, 1 = no-fly
        # ----------------------------------------------------
        f.write("SCALARS category int 1\n")
        f.write("LOOKUP_TABLE default\n")

        for _z in zs:
            for iy in range(ny):
                for ix in range(nx):
                    f.write(f"{int(category2d[iy, ix])}\n")

    return npoints

def save_vtk_model_nodes(output_file, xs, ys, zs, slow2d, mask2d):
    """
    Save model nodes as legacy ASCII VTK POLYDATA.

    This file is for visualizing model nodes as points in ParaView.

    Coordinates:
        x = xs, local model x in meters
        y = ys, local model y in meters
        z = zs, elevation in meters

    Point data:
        slowness_s_per_m
        flyable_mask
        category

    Category:
        0 = flyable
        1 = no-fly
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    nx = len(xs)
    ny = len(ys)
    nz = len(zs)
    npoints = nx * ny * nz

    category2d = np.where(mask2d == 1, 0, 1).astype(int)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Hoa Lac Scenario 1 model nodes\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # ----------------------------------------------------
        # Points
        # VTK point order: x fastest, then y, then z.
        # Same order as raw.xyz: z -> y -> x.
        # ----------------------------------------------------
        f.write(f"POINTS {npoints} float\n")

        for z in zs:
            for y in ys:
                for x in xs:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        # ----------------------------------------------------
        # Vertices: one vertex cell for each point.
        # Format:
        #   VERTICES npoints npoints*2
        #   1 point_id
        # ----------------------------------------------------
        f.write(f"VERTICES {npoints} {npoints * 2}\n")

        for pid in range(npoints):
            f.write(f"1 {pid}\n")

        f.write(f"POINT_DATA {npoints}\n")

        # Slowness
        f.write("SCALARS slowness_s_per_m float 1\n")
        f.write("LOOKUP_TABLE default\n")

        for _z in zs:
            for iy in range(ny):
                for ix in range(nx):
                    f.write(f"{slow2d[iy, ix]:.8e}\n")

        # Flyable mask
        f.write("SCALARS flyable_mask int 1\n")
        f.write("LOOKUP_TABLE default\n")

        for _z in zs:
            for iy in range(ny):
                for ix in range(nx):
                    f.write(f"{int(mask2d[iy, ix])}\n")

        # Category
        f.write("SCALARS category int 1\n")
        f.write("LOOKUP_TABLE default\n")

        for _z in zs:
            for iy in range(ny):
                for ix in range(nx):
                    f.write(f"{int(category2d[iy, ix])}\n")

    return npoints

def save_vtk_polygon_cage(output_file, poly, origin_xy, zmin, zmax):
    """
    Save AOI polygon cage as legacy ASCII VTK POLYDATA.

    Output contains:
        1. AOI polygon at minimum elevation
        2. AOI polygon at maximum elevation
        3. Vertical lines connecting each polygon vertex from zmin to zmax

    Coordinates are local model coordinates, same as raw.xyz:
        x = UTM x - origin_x
        y = UTM y - origin_y
        z = elevation
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    x0, y0 = origin_xy

    coords = np.asarray(poly.exterior.coords)

    # Remove duplicated closing point for vertical vertices.
    coords_open = coords[:-1]

    px = coords_open[:, 0] - x0
    py = coords_open[:, 1] - y0

    nvert = len(px)

    # Points:
    #   first nvert points  = bottom polygon
    #   second nvert points = top polygon
    points = []

    for x, y in zip(px, py):
        points.append((x, y, zmin))

    for x, y in zip(px, py):
        points.append((x, y, zmax))

    npoints = len(points)

    # Lines:
    #   bottom closed polygon = nvert + 1 ids
    #   top closed polygon    = nvert + 1 ids
    #   vertical lines        = nvert lines, each with 2 ids
    lines = []

    bottom_ids = list(range(nvert)) + [0]
    top_ids = list(range(nvert, 2 * nvert)) + [nvert]

    lines.append(bottom_ids)
    lines.append(top_ids)

    for i in range(nvert):
        lines.append([i, i + nvert])

    nlines = len(lines)
    line_size = sum(len(line) + 1 for line in lines)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Hoa Lac AOI polygon cage\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        f.write(f"POINTS {npoints} float\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        f.write(f"LINES {nlines} {line_size}\n")
        for line in lines:
            f.write(f"{len(line)} " + " ".join(str(i) for i in line) + "\n")

        # Point data: mark bottom/top vertices
        # 0 = bottom, 1 = top
        f.write(f"POINT_DATA {npoints}\n")
        f.write("SCALARS polygon_level int 1\n")
        f.write("LOOKUP_TABLE default\n")

        for _ in range(nvert):
            f.write("0\n")

        for _ in range(nvert):
            f.write("1\n")

        # Cell data: mark line type
        # 0 = bottom polygon
        # 1 = top polygon
        # 2 = vertical connector
        f.write(f"CELL_DATA {nlines}\n")
        f.write("SCALARS line_type int 1\n")
        f.write("LOOKUP_TABLE default\n")

        f.write("0\n")
        f.write("1\n")

        for _ in range(nvert):
            f.write("2\n")

    return npoints, nlines

# ============================================================
# PyGMT plotting
# ============================================================

def polygon_to_local_xy(poly, origin_xy):
    x0, y0 = origin_xy
    coords = np.asarray(poly.exterior.coords)
    px = coords[:, 0] - x0
    py = coords[:, 1] - y0
    return px, py


def get_region_from_polygon(poly, padding=REGION_PADDING):
    """
    Region from polygon lon/lat bounds + padding only.
    No BBOX is used.
    """
    lon, lat = polygon_to_lonlat(poly)

    return [
        lon.min() - padding,
        lon.max() + padding,
        lat.min() - padding,
        lat.max() + padding,
    ]


def get_surface_spacing_deg():
    """
    Surface spacing in degree.
    If SURFACE_SPACING_DEG is set, use it.
    Otherwise estimate from DX/DY in meters.
    """
    if SURFACE_SPACING_DEG is not None:
        return SURFACE_SPACING_DEG

    return min(DX, DY) / 111320.0


def make_region_compatible_with_spacing(region, spacing):
    """
    Expand east/north slightly so GMT surface accepts the region.

    GMT surface requires:
        xmax - xmin = NX * spacing
        ymax - ymin = NY * spacing
    """
    west, east, south, north = region

    nx = int(np.ceil((east - west) / spacing))
    ny = int(np.ceil((north - south) / spacing))

    east_new = west + nx * spacing
    north_new = south + ny * spacing

    return [west, east_new, south, north_new]


def make_2class_cpt(cpt_file):
    """
    Make categorical CPT for:
        0 = Flyable
        1 = No-fly
    """
    cpt_file = Path(cpt_file)

    pygmt.makecpt(
        cmap="categorical",
        series=(0, 1, 1),
        color_model="+cFlyable,No-fly",
        continuous=False,
        output=str(cpt_file),
    )

    return cpt_file


def plot_2d_z0_categorical(fig_dir, xs_utm, ys_utm, mask2d, poly):
    """
    2D geographic map at z = 0 m.

    Region:
        polygon bounds + REGION_PADDING only.
        Do not use BBOX.

    Category:
        0 = Flyable
        1 = No-fly
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cpt_file = make_2class_cpt(fig_dir / "category_flyable.cpt")

    # Convert model nodes from UTM to lon/lat.
    xx_utm, yy_utm = np.meshgrid(xs_utm, ys_utm)
    lon_flat, lat_flat = utm_to_lonlat(xx_utm.ravel(), yy_utm.ravel())

    # Categorical code:
    # 0 = flyable, 1 = no-fly.
    cat2d = np.where(mask2d == 1, 0, 1)
    cat_flat = cat2d.ravel().astype(int)

    # Polygon in lon/lat.
    plon, plat = polygon_to_lonlat(poly)

    # Region from polygon + REGION_PADDING only.
    region0 = get_region_from_polygon(poly, padding=REGION_PADDING)

    # Surface spacing.
    spacing = get_surface_spacing_deg()

    # Make region compatible with GMT surface.
    region = make_region_compatible_with_spacing(region0, spacing)

    # Temporary XYZ and grid.
    xyz_file = fig_dir / "_tmp_z0_category_lonlat.xyz"
    grid_file = fig_dir / "_tmp_z0_category_surface.nc"
    legend_file = fig_dir / "_tmp_legend_2d.txt"

    np.savetxt(
        xyz_file,
        np.column_stack([lon_flat, lat_flat, cat_flat]),
        fmt="%.8f %.8f %d",
    )

    # Surface categorical field.
    pygmt.surface(
        data=str(xyz_file),
        region=region,
        spacing=spacing,
        outgrid=str(grid_file),
    )

    # Clip continuous surface back to 2 classes.
    # < 0.5  -> 0 = Flyable
    # >= 0.5 -> 1 = No-fly
    grid_cat = pygmt.grdclip(
        grid=str(grid_file),
        below=[0.5, 0],
        above=[0.5, 1],
    )

    title = f'"{PLACE_NAME} AOI"' if USE_PLACE_NAME else '"Hoa Lac study-area AOI"'

    fig = pygmt.Figure()

    fig.basemap(
        region=region,
        projection=PROJECTION,
        frame=[
            "xaf",
            "yaf",
            f"WSen+t{title}",
        ],
    )

    # Filled categorical region.
    fig.grdimage(
        grid=grid_cat,
        cmap=str(cpt_file),
        transparency=CATEGORY_TRANSPARENCY,
    )

    # Purple polygon frame.
    fig.plot(
        x=plon,
        y=plat,
        pen=POLYGON_PEN_2D,
    )

    # Black model nodes.
    fig.plot(
        x=lon_flat,
        y=lat_flat,
        style=DOT_SIZE_2D,
        fill="black",
        pen=None,
    )

    # Legend only for polygon and nodes.
    with open(legend_file, "w", encoding="utf-8") as f:
        f.write("G 0.08c\n")
        f.write("S 0.25c c 0.08c black - 0.55c Model nodes\n")
        f.write(f"S 0.25c - 0.45c - 1.2p,{POLYGON_RGB} 0.55c AOI polygon\n")

    fig.legend(
        spec=str(legend_file),
        position="JBL+jBL+o0.3c/0.3c",
        box="+gwhite+p0.8p,black",
    )

    # Colorbar for categorical classes.
    fig.colorbar(
        cmap=str(cpt_file),
        position="JBC+w6c/0.35c+o0.0c/1c+h",
        frame="xaf+lCategory",
    )

    out_png = fig_dir / "model_2d_z0_categorical.png"
    fig.savefig(out_png, dpi=DPI)
    print(f"[OK] Saved figure: {out_png}")

    # Cleanup.
    if xyz_file.exists():
        xyz_file.unlink()

    if legend_file.exists():
        legend_file.unlink()


def plot_3d_nodes_perspective(fig_dir, xs, ys, zs, mask2d, poly, origin_xy):
    """
    3D perspective view of model nodes.

    - Categorical colors from PyGMT CPT.
    - Purple polygon at min and max elevation.
    - Purple vertical lines connecting polygon vertices.
    - Bottom frame.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cpt_file = make_2class_cpt(fig_dir / "category_flyable.cpt")

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="xy")

    cat2d = np.where(mask2d == 1, 0, 1)
    cat3d = np.repeat(cat2d[:, :, np.newaxis], len(zs), axis=2)

    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = zz.ravel()
    cat_flat = cat3d.ravel().astype(int)

    region = [
        xs.min(),
        xs.max(),
        ys.min(),
        ys.max(),
        zs.min(),
        zs.max(),
    ]

    px, py = polygon_to_local_xy(poly, origin_xy)

    zmin = zs.min()
    zmax = zs.max()

    fig = pygmt.Figure()

    fig.basemap(
        region=region,
        projection=FIG_3D_PROJECTION,
        zsize=FIG_3D_ZSIZE,
        perspective=FIG_3D_PERSPECTIVE,
        frame=[
            "xaf+lX (m)",
            "yaf+lY (m)",
            "zaf+lZ elevation (m)",
            "WSenZ+b+tScenario 1 no velocity: 3D model nodes",
        ],
    )

    fig.plot3d(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        fill=cat_flat,
        cmap=str(cpt_file),
        style=DOT_SIZE_3D,
        pen=None,
        transparency=CATEGORY_TRANSPARENCY,
        perspective=True,
    )

    # Bottom rectangular frame.
    bx = [xs.min(), xs.max(), xs.max(), xs.min(), xs.min()]
    by = [ys.min(), ys.min(), ys.max(), ys.max(), ys.min()]
    bz = [zmin, zmin, zmin, zmin, zmin]

    fig.plot3d(
        x=bx,
        y=by,
        z=bz,
        pen="1.2p,black",
        perspective=True,
    )

    # Polygon at minimum elevation.
    fig.plot3d(
        x=px,
        y=py,
        z=np.full_like(px, zmin, dtype=float),
        pen=POLYGON_PEN_3D,
        perspective=True,
    )

    # Polygon at maximum elevation.
    fig.plot3d(
        x=px,
        y=py,
        z=np.full_like(px, zmax, dtype=float),
        pen=POLYGON_PEN_3D,
        perspective=True,
    )

    # Vertical connections between polygon vertices.
    for vx, vy in zip(px[:-1], py[:-1]):
        fig.plot3d(
            x=[vx, vx],
            y=[vy, vy],
            z=[zmin, zmax],
            pen=VERTEX_CONNECT_PEN_3D,
            perspective=True,
        )

    fig.colorbar(
        cmap=str(cpt_file),
        position="JBC+w8c/0.35c+h+o0c/1.0c",
        frame="xaf+lCategory",
    )

    out_png = fig_dir / "model_3d_nodes_perspective.png"
    fig.savefig(out_png, dpi=DPI)
    print(f"[OK] Saved figure: {out_png}")


def plot_all_figures(fig_dir, xs, ys, xs_utm, ys_utm, zs, mask2d, poly, origin_xy):
    """
    Only two figures:
        1. 2D z = 0 m categorical geographic map
        2. 3D perspective model nodes
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\nPlotting two figures with PyGMT...")

    plot_2d_z0_categorical(
        fig_dir=fig_dir,
        xs_utm=xs_utm,
        ys_utm=ys_utm,
        mask2d=mask2d,
        poly=poly,
    )

    plot_3d_nodes_perspective(
        fig_dir=fig_dir,
        xs=xs,
        ys=ys,
        zs=zs,
        mask2d=mask2d,
        poly=poly,
        origin_xy=origin_xy,
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 72)
    print("Create nested-base model: Scenario 1 no velocity")
    print("=" * 72)

    poly, poly_source = load_area_polygon()

    print("\nPolygon info:")
    print(f"  area   = {poly.area:.3f} m2")
    print(f"  bounds = {poly.bounds}")
    print(f"  source = {poly_source}")

    xs, ys, xs_utm, ys_utm, zs, slow2d, mask2d, origin_xy = make_grid_values(
        poly=poly,
        dx=DX,
        dy=DY,
        dz=DZ,
        zmin=ZMIN,
        zmax=ZMAX,
        inside_slow=INSIDE_SLOWNESS,
        outside_slow=OUTSIDE_SLOWNESS,
        origin_mode=ORIGIN_MODE,
    )

    n = save_raw_xyz(
        output_file=OUTPUT_XYZ,
        xs_utm=xs_utm,
        ys_utm=ys_utm,
        zs=zs,
        slow2d=slow2d,
    )

    n_vtk = save_vtk_rectilinear_grid(
        output_file=OUTPUT_VTK,
        xs=xs,
        ys=ys,
        zs=zs,
        slow2d=slow2d,
        mask2d=mask2d,
    )

    n_nodes_vtk = save_vtk_model_nodes(
        output_file=OUTPUT_NODES_VTK,
        xs=xs,
        ys=ys,
        zs=zs,
        slow2d=slow2d,
        mask2d=mask2d,
    )

    n_poly_points, n_poly_lines = save_vtk_polygon_cage(
        output_file=OUTPUT_POLYGON_VTK,
        poly=poly,
        origin_xy=origin_xy,
        zmin=ZMIN,
        zmax=ZMAX,
    )

    print("\nOutput model:")
    print(f"  output xyz         = {OUTPUT_XYZ}")
    print(f"  output vtk grid    = {OUTPUT_VTK}")
    print(f"  output vtk nodes   = {OUTPUT_NODES_VTK}")
    print(f"  output vtk polygon = {OUTPUT_POLYGON_VTK}")
    print(f"  xyz points         = {n:,}")
    print(f"  vtk grid points    = {n_vtk:,}")
    print(f"  vtk node points    = {n_nodes_vtk:,}")
    print(f"  polygon points     = {n_poly_points:,}")
    print(f"  polygon lines      = {n_poly_lines:,}")
    print(f"  origin x0/y0 = {origin_xy[0]:.3f}, {origin_xy[1]:.3f}")
    print(f"  z range      = {ZMIN} to {ZMAX} m")
    print(f"  dx dy dz     = {DX} {DY} {DZ} m")
    print(f"  inside       = {INSIDE_SLOWNESS} s/m")
    print(f"  outside      = {OUTSIDE_SLOWNESS} s/m")

    if DO_PLOT:
        plot_all_figures(
            fig_dir=FIG_DIR,
            xs=xs,
            ys=ys,
            xs_utm=xs_utm,
            ys_utm=ys_utm,
            zs=zs,
            mask2d=mask2d,
            poly=poly,
            origin_xy=origin_xy,
        )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
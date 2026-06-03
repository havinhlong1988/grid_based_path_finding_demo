#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare input data for scenario 1 without velocity.

Copies XYZ data for:
  - roads
  - buildings
  - DEM / topography
  - OSM extra features
  - KML/KMZ plan exported points/rings

Output:
  input/02_data_senario1_no_velocity/
"""

from pathlib import Path
import shutil


# ============================================================
# User settings
# ============================================================

PROJECT_DIR = Path(".").resolve()

OUT_DIR = PROJECT_DIR / "input" / "02_data_senario1_no_velocity"

# Keep categories separated for easy plotting later
COPY_RULES = {
    "roads": [
        "output*/**/*road*.xyz",
        "output*/**/*roads*.xyz",
        "input*/**/*road*.xyz",
        "input*/**/*roads*.xyz",
    ],

    "buildings": [
        "output*/**/*building*.xyz",
        "output*/**/*buildings*.xyz",
        "output*/**/*obm*.xyz",
        "input*/**/*building*.xyz",
        "input*/**/*buildings*.xyz",
    ],

    "dem": [
        "output*/**/*dem*.xyz",
        "output*/**/*DEM*.xyz",
        "output*/**/*topo*.xyz",
        "output*/**/*topography*.xyz",
        "input*/**/*dem*.xyz",
        "input*/**/*DEM*.xyz",
        "input*/**/*topo*.xyz",
        "input*/**/*topography*.xyz",
    ],

    "osm_extra_features": [
        "output*/**/*osm*.xyz",
        "output*/**/*OSM*.xyz",
        "output*/**/*feature*.xyz",
        "output*/**/*extra*.xyz",
        "input*/**/*osm*.xyz",
        "input*/**/*OSM*.xyz",
        "input*/**/*feature*.xyz",
        "input*/**/*extra*.xyz",
    ],

    "kml_plan": [
        "output/01_HoaLac_studies_area/kml_plan/*.xyz",
        "output*/**/kml_plan/*.xyz",
        "kml/plan/*.xyz",
    ],
}

# Avoid copying files from the target directory back into itself
EXCLUDE_DIRS = {
    OUT_DIR.resolve(),
}


# ============================================================
# Functions
# ============================================================

def is_inside(child: Path, parent: Path) -> bool:
    """Return True if child is inside parent."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def collect_files(patterns):
    """Collect unique XYZ files from glob patterns."""
    files = []

    for pattern in patterns:
        for f in PROJECT_DIR.glob(pattern):
            if not f.is_file():
                continue

            if f.suffix.lower() != ".xyz":
                continue

            # skip target output directory
            if any(is_inside(f, exdir) for exdir in EXCLUDE_DIRS):
                continue

            files.append(f.resolve())

    # remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)

    return unique_files


def safe_copy(src: Path, dst_dir: Path) -> Path:
    """
    Copy src to dst_dir.
    If filename already exists, append _001, _002, ...
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / src.name

    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    stem = src.stem
    suffix = src.suffix

    i = 1
    while True:
        new_dst = dst_dir / f"{stem}_{i:03d}{suffix}"
        if not new_dst.exists():
            shutil.copy2(src, new_dst)
            return new_dst
        i += 1


def write_manifest(records):
    """Write a simple manifest CSV."""
    manifest = OUT_DIR / "manifest_copied_xyz_files.csv"

    with open(manifest, "w", encoding="utf-8") as f:
        f.write("category,source_file,copied_file\n")
        for category, src, dst in records:
            f.write(f"{category},{src},{dst}\n")

    return manifest


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    print("=" * 70)
    print("Preparing data for scenario 1 without velocity")
    print(f"Project dir : {PROJECT_DIR}")
    print(f"Output dir  : {OUT_DIR}")
    print("=" * 70)

    for category, patterns in COPY_RULES.items():
        dst_category_dir = OUT_DIR / category
        files = collect_files(patterns)

        print(f"\n[{category}]")
        print(f"Found {len(files)} XYZ file(s)")

        if len(files) == 0:
            print("  WARNING: no file found")
            continue

        for src in files:
            dst = safe_copy(src, dst_category_dir)
            records.append((category, src, dst))
            print(f"  copied: {src.relative_to(PROJECT_DIR)}")
            print(f"       -> {dst.relative_to(PROJECT_DIR)}")

    manifest = write_manifest(records)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Copied XYZ files : {len(records)}")
    print(f"Output folder    : {OUT_DIR}")
    print(f"Manifest         : {manifest}")
    print("=" * 70)


if __name__ == "__main__":
    main()
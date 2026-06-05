from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_FILE = PROJECT_DIR / "model" / "senario1" / "model_senario1_with_label.xyz"

# ============================================================
# Output settings
# ============================================================

DAT_ROOT_DIR = PROJECT_DIR / "output" / "dat" / "senario1"
FIGURE_ROOT_DIR = PROJECT_DIR / "output" / "figures" / "senario1"

PATH_NAME = "path_senario1"

# Algorithm options:
#   "astar"
#   "flood_fill"
ALGORITHM = "astar"

START_LABEL = "DB01"
END_LABEL = "DK01"

START_COORD = None
END_COORD = None

BLOCK_LABEL_PREFIXES = ("RA",)
HIGH_COST_LABEL_PREFIXES = ("FLZ",)

FLZ_COST_FACTOR = 5.0

CONNECTIVITY_2D = 8
CONNECTIVITY_3D = 26
# ============================================================
# Force important nodes to be flyable
# ============================================================

# DB = Drone base
# DK = Docking / landing point
# These nodes must remain flyable even if their slowness or area says no-fly.
ALWAYS_FLYABLE_PREFIXES = ("DB", "DK")

# If True, also force the snapped search start/end grid nodes to be flyable.
# This is important when DB/DK are special KML points that are snapped to nearest grid node.
FORCE_SEARCH_START_END_FLYABLE = True

# ============================================================
# Start/end snapping
# ============================================================

# DB/DK points may not lie exactly on the regular path-finding grid.
# If True, the algorithm uses the nearest traversable grid node,
# while the exported/plot path still starts at DB and ends at DK.
SNAP_START_END_TO_GRID = True

# Only snap to these node types.
# Usually avoid DB/DK/RA.
SNAP_TARGET_PREFIXES = ("N", "FLZ")

# If True, add original DB/DK points to exported path footprint.
INCLUDE_REAL_START_END_IN_OUTPUT = True

# Plot settings
PLOT_MAX_MODEL_POINTS = 300000
PLOT_DPI = 300
PLOT_MODEL_ALPHA = 0.45
PLOT_MODEL_MARKER_SIZE = 1.0
PLOT_PATH_LINE_WIDTH = 1.0
PLOT_REPORT_TEXT_BOX = True
# ============================================================
# Initiate plot settings
# ============================================================

PLOT_INITIATE_FIGURE = True
INITIATE_FIGURE_NAME = f"00_initiate_from_{START_LABEL}_to_{END_LABEL}.png"
# ============================================================
# Plot classification by slowness / flyability
# ============================================================

# If True, model map is shown as 2 categories:
#   flyable
#   no-fly
PLOT_MODEL_AS_FLYABLE_NOFLY = True

PLOT_ALWAYS_FLYABLE_PREFIXES = ("DB", "DK")
# ============================================================
# Endpoint flyable buffer
# ============================================================

# Open a small flyable buffer around start/end for takeoff and landing.
# Unit:
#   - meters if x/y are projected coordinates
#   - automatically converted approximately to degrees if x/y are lon/lat
ENDPOINT_FLYABLE_BUFFER_RADIUS_M = 100.0

# Apply buffer around:
#   "real"    = DB/DK coordinates
#   "search"  = snapped search start/end grid nodes
#   "both"    = both DB/DK and snapped grid nodes
ENDPOINT_FLYABLE_BUFFER_MODE = "both"

# No-fly if label_prefix is in this list
PLOT_NO_FLY_PREFIXES = ("RA",)

# Also no-fly if slowness >= this threshold
# Adjust based on your model values.
# Example:
#   flyable nodes ~ 0.0 to 1.0
#   blocked nodes ~ 1e5 or 1e6
PLOT_NO_FLY_SLOWNESS_THRESHOLD = 10

# Optional: show FLZ nodes on top as separate overlay
PLOT_SHOW_FLZ_OVERLAY = True
# ============================================================
# Slowness cap after model loading
# ============================================================

CAP_SLOWNESS_AFTER_LOAD = True

# Usually use the same threshold as no-fly plotting.
# Example: all slowness > 1e5 becomes 1e5.
SLOWNESS_CAP_VALUE = PLOT_NO_FLY_SLOWNESS_THRESHOLD

# ============================================================
# Graph neighbor construction
# ============================================================

# Better for lon/lat model nodes.
GRAPH_NEIGHBOR_MODE = "kdtree"

# Radius multiplier relative to estimated grid spacing.
# For 8-connectivity 2D, 1.45–1.60 is good.
KDTREE_RADIUS_FACTOR = 1.60

# Limit neighbors for speed.
# 2D:
#   4 or 8
# 3D:
#   6, 18, or 26
KDTREE_MAX_NEIGHBORS_2D = 8
KDTREE_MAX_NEIGHBORS_3D = 26
# ============================================================
# Cleanup settings
# ============================================================

RUN_CLEANUP = True
CLEANUP_DRY_RUN = False
CLEANUP_EMPTY_DIRS = True

CLEANUP_TARGET_DIRS = [
    DAT_ROOT_DIR,
    FIGURE_ROOT_DIR,
]

CLEANUP_PATTERNS = [
    "*.tmp",
    "*.temp",
    "*.bak",
    "*.backup",
    "*.log",
    "*.cache",
    "*.gmt",
    "*.cpt",
    "*.grd",
    "*.nc",
    "*.vrt",
    "*.aux.xml",
    "*_tmp.*",
    "*_temp.*",
    "tmp_*",
    "temp_*",
    ".gmt*",
]
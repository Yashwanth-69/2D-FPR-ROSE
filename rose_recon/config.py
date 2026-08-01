"""Tuning constants and the per-run output paths.

Two kinds of setting live here and they behave differently on purpose.

**Tuning constants** are plain module-level values, exported through ``__all__``
so the stage modules can pull them in with ``from .config import *``. They never
change during a run.

**Paths** depend on which video is being processed, so they start as ``None``
and are filled in by :func:`configure`. They are deliberately kept OUT of
``__all__``: a stage module must reach them through ``config.WALL_NPY`` rather
than importing the bare name, because a bare import would capture ``None`` at
import time and then silently write to the wrong place. Excluding them means any
reference that forgets the ``config.`` prefix raises NameError immediately
instead of quietly misbehaving.
"""

import os
import sys

__all__ = [
    # stage 3 -- the ROSE spectral filter
    "N_DIRECTIONS", "LOW_FREQ_KEEP", "WEDGE_DEG", "KEEP_FRACTION",
    "INT_WIN", "INT_Q",
    # runtime behaviour
    "SAVE_FRAMES", "SHOW_ODOM", "SHOW_PLOTS", "FUSE_STAGES",
    # inherited casualty-detection settings (disabled, see DETECT_VICTIMS)
    "DETECT_VICTIMS", "VICTIM_CLASSES", "VICTIM_CONF", "VICTIM_MERGE_RADIUS_M",
    "CLUSTER_RADIUS_M", "MIN_DETECTION_HITS", "LAUNCH_EXCLUDE_M",
    "EXPECTED_VICTIM_COUNT", "USE_BOX_FOOTPRINT", "ASSUMED_BODY_H_M",
    "BODY_H_SIGMA_M", "HUBER_DELTA_M", "IRLS_ITERS", "NADIR_KEEP_FRAC",
    "NADIR_KEEP_MIN", "PARALLAX_MIN_VAR", "TRI_MAX_Z_M", "TRI_TRUST_M",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)

# ---- STAGE 3: ROSE spectral filter ------------------------------------------
# Frozen values. N_DIRECTIONS is how many angular ridges count as "wall
# directions" (2 for a rectilinear building); WEDGE_DEG is the half-width kept
# around each; LOW_FREQ_KEEP passes the lowest frequencies through untouched so
# the building's overall shape survives; KEEP_FRACTION and INT_Q are the
# structure and intensity cuts that follow the inverse transform.
N_DIRECTIONS = 2
LOW_FREQ_KEEP = 3
WEDGE_DEG = 12.9
KEEP_FRACTION = 0.84
INT_WIN = 3
INT_Q = 0.44

# ---- runtime ----------------------------------------------------------------
SAVE_FRAMES = False      # write captured_frames/*.jpg
SHOW_ODOM = True         # live stage-1 trajectory windows
SHOW_PLOTS = False       # stage-1 matplotlib windows (the figure is saved anyway)
FUSE_STAGES = True       # odometry and wall grid share ONE video pass

# ---- casualty detection: OFF -------------------------------------------------
# The stage-1 code this project inherits can also detect people on the same video
# pass. That is irrelevant to wall reconstruction and it is what would pull in the
# person-detection weights, so it is disabled and that model is never loaded. The
# constants below are still referenced by the inherited code paths.
DETECT_VICTIMS = False
VICTIM_CLASSES = [0]
VICTIM_CONF = 0.60
VICTIM_MERGE_RADIUS_M = 1.5
CLUSTER_RADIUS_M = 1.5
MIN_DETECTION_HITS = 40
LAUNCH_EXCLUDE_M = 1.5
EXPECTED_VICTIM_COUNT = None
USE_BOX_FOOTPRINT = True
ASSUMED_BODY_H_M = 0.55
BODY_H_SIGMA_M = 0.35
HUBER_DELTA_M = 0.40
IRLS_ITERS = 6
NADIR_KEEP_FRAC = 0.30
NADIR_KEEP_MIN = 8
PARALLAX_MIN_VAR = 0.020
TRI_MAX_Z_M = 3.00
TRI_TRUST_M = 1.50

# ---- per-run paths, set by configure() --------------------------------------
# Not exported: reach these as config.NAME so a missed reference fails loudly.
VIDEO_PATH = None        # flyover footage being reconstructed
IMU_CSV = None           # its inertial log
OUT_DIR = None           # everything this run writes
MAPPING_DIR = None       # working set (same as OUT_DIR)
CSV_PATH = None          # recovered trajectory
WALL_NPY = None          # raw wall accumulation
TUNING_DIR = None        # stage-3 intermediates
FRAMES_DIR = None        # captured frames, when SAVE_FRAMES
BEFORE_PNG = None        # raw accumulation, as an image
ROSE_PNG = None          # after the spectral filter -- the deliverable
OUT_PATH = None          # alias of ROSE_PNG, used by inherited stage-2 helpers
MODEL_PATH = None        # wall/floor segmentation weights
VICTIM_MODEL_PATH = None       # never loaded while DETECT_VICTIMS is False
VICTIM_OUT_CSV = None
VICTIM_UNC_CSV = None

FORCE_ODOMETRY = False
FORCE_MAPPING = False


def configure(video, imu=None, out=None, force=False, force_map=False,
              show=True, model=None):
    """Resolve every per-run path from the video being processed.

    Results are cached per video, so two flyovers never overwrite each other's
    trajectory or wall grid.

    Args:
        video: flyover footage to reconstruct. Required.
        imu: inertial log. Defaults to the video path with a ``.csv`` extension,
            which is how recordings are normally shipped.
        out: output folder. Defaults to ``output/<video name>/``.
        force: redo the video pass even if a trajectory is cached.
        force_map: redo the wall accumulation even if a grid is cached.
        show: display the live stage-1 trajectory windows.
        model: wall/floor segmentation weights. Defaults to the bundled model.

    Returns:
        This module, so callers can read the resolved paths straight back.
    """
    global VIDEO_PATH, IMU_CSV, OUT_DIR, MAPPING_DIR, CSV_PATH, WALL_NPY
    global TUNING_DIR, FRAMES_DIR, BEFORE_PNG, ROSE_PNG, OUT_PATH, MODEL_PATH
    global VICTIM_MODEL_PATH, VICTIM_OUT_CSV, VICTIM_UNC_CSV
    global FORCE_ODOMETRY, FORCE_MAPPING, SHOW_ODOM

    VIDEO_PATH = os.path.abspath(video)
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError("video not found: %s" % VIDEO_PATH)
    IMU_CSV = os.path.abspath(imu or os.path.splitext(VIDEO_PATH)[0] + ".csv")

    stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    OUT_DIR = os.path.abspath(out or os.path.join(_PROJECT, "output", stem))
    MAPPING_DIR = OUT_DIR
    CSV_PATH = os.path.join(OUT_DIR, "camera_poses.csv")
    WALL_NPY = os.path.join(OUT_DIR, "wall_raw.npy")
    TUNING_DIR = os.path.join(OUT_DIR, "tuning")
    FRAMES_DIR = os.path.join(OUT_DIR, "captured_frames")
    BEFORE_PNG = os.path.join(OUT_DIR, "walls_before.png")
    ROSE_PNG = os.path.join(OUT_DIR, "walls_rose.png")
    OUT_PATH = ROSE_PNG

    MODEL_PATH = os.path.abspath(
        model or os.path.join(_PROJECT, "models", "best_segmentation.pt"))
    VICTIM_MODEL_PATH = ""
    VICTIM_OUT_CSV = os.path.join(OUT_DIR, "victims_unused.csv")
    VICTIM_UNC_CSV = os.path.join(OUT_DIR, "victims_unused_uncertainty.csv")

    FORCE_ODOMETRY = bool(force)
    FORCE_MAPPING = bool(force_map)
    SHOW_ODOM = bool(show)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TUNING_DIR, exist_ok=True)
    return sys.modules[__name__]


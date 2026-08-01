"""Runs the three stages in order and writes the before/after pair.

Stage 1 and stage 2 are cached per video, so a re-run only redoes stage 3, which
takes about a second. That is deliberate: the spectral filter is the part worth
iterating on, and it should never cost a video pass to try a different setting.
"""

import math
import os

import cv2
import numpy as np

from . import config
from .mapping import WallMapper, run_mapping
from .odometry import run_odometry
from .rose import rose_intensity


def save_before_after(raw, binary):
    """Write the pair this project exists to produce.

    Both are white-on-black at the grid's own resolution, so they line up pixel
    for pixel and can be diffed directly.
    """
    cv2.imwrite(config.BEFORE_PNG, (raw > 0).astype(np.uint8) * 255)
    cv2.imwrite(config.ROSE_PNG, (binary > 0).astype(np.uint8) * 255)
    before_px = int((raw > 0).sum())
    after_px = int((binary > 0).sum())
    drop = (1.0 - after_px / before_px) * 100.0 if before_px else 0.0
    print("\n  before : %-7d occupied cells  -> %s" % (before_px, config.BEFORE_PNG))
    print("  after  : %-7d occupied cells  -> %s" % (after_px, config.ROSE_PNG))
    print("  removed: %.1f%% of the occupancy as non-structural" % drop)
    return before_px, after_px


def reconstruct():
    """Reconstruct the wall map for the video already passed to configure().

    Returns:
        (binary, wall_dirs): the denoised occupancy map, and the dominant wall
        directions the filter locked onto, in radians.
    """
    if config.VIDEO_PATH is None:
        raise RuntimeError("call rose_recon.configure(video=...) first")
    print("========== 2D RECONSTRUCTION FROM 3D FLYOVER ==========")
    print("  video  : %s" % config.VIDEO_PATH)
    print("  imu    : %s%s" % (config.IMU_CSV,
                               "" if os.path.exists(config.IMU_CSV)
                               else "   (missing)"))
    print("  output : %s" % config.OUT_DIR)

    if not os.path.exists(config.MODEL_PATH):
        raise SystemExit(
            "segmentation model not found: %s\n"
            "Place best_segmentation.pt in the models/ folder."
            % config.MODEL_PATH)

    need_poses = config.FORCE_ODOMETRY or not os.path.exists(config.CSV_PATH)
    need_map = config.FORCE_MAPPING or not os.path.exists(config.WALL_NPY)

    if need_poses or need_map:
        if config.FUSE_STAGES:
            # One video pass feeds both the trajectory and the wall grid, which
            # halves the decoding and inference work.
            print("\n========== STAGE 1+2: odometry + wall grid ==========")
            run_odometry(config.VIDEO_PATH, config.IMU_CSV, mapper=WallMapper())
        else:
            if need_poses:
                print("\n========== STAGE 1: visual odometry ==========")
                run_odometry(config.VIDEO_PATH, config.IMU_CSV)
            if need_map:
                print("\n========== STAGE 2: wall grid from frames ==========")
                run_mapping()
    else:
        print("\n[cache] trajectory and wall grid already built for this video;"
              " re-running stage 3 only.")
        print("        (force=True redoes the video pass, force_map=True the grid)")

    if not os.path.exists(config.WALL_NPY):
        raise SystemExit("stage 2 produced no wall grid (%s)" % config.WALL_NPY)

    print("\n========== STAGE 3: ROSE spectral filter ==========")
    raw = (np.load(config.WALL_NPY).astype(np.uint8)) * 255
    binary, wall_dirs = rose_intensity(raw)
    print("  dominant wall directions: %s"
          % ", ".join("%.1f deg" % math.degrees(d) for d in wall_dirs))
    save_before_after(raw, binary)
    print("\ndone.")
    return binary, wall_dirs

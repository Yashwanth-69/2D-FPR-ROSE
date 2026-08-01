"""Shared perception: model loading and the segmentation masks.

Both stages need these, so they live here rather than in either one. Stage 1
uses the floor mask to track the ground plane; stage 2 uses the wall mask to
stamp occupancy. Keeping them together is also what breaks the import cycle the
two stages would otherwise form.

DEVICE is module state on purpose: pick_device() resolves it once and caches it
here, so it must not be re-exported into other modules.
"""

import numpy as np
import cv2

from . import config
from .config import *  # noqa: F401,F403

# Resolved once by pick_device() and cached here. It lives in this module rather
# than in config because pick_device() ASSIGNS it: a value that gets mutated
# cannot be re-exported through `from .config import *`, since every importer
# would bind its own copy and never see the update.
DEVICE = None

# Mask tuning. These live here with the functions that use them; they were left
# behind in the stage modules when these helpers moved, which is why the split
# broke. Thresholds are deliberately loose -- the spectral filter in stage 3 is
# what removes false positives, so the segmenter is allowed to over-call walls.
WALL_SUSPICION_CONF = 0.20   # loose: anything suspected wall is banned
WALL_DILATE_PX      = 10     # safety margin around wall masks (px)
SEG_CONF            = 0.20   # wall mask confidence
MASK_OPEN           = 3      # despeckle the segmentation mask before rays

def pick_device():
    """cuda if torch sees a GPU, else cpu. Called once, result cached."""
    global DEVICE
    if DEVICE is None:
        try:
            import torch
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            if DEVICE == "cuda":
                print("[gpu] CUDA available: %s"
                      % torch.cuda.get_device_name(0))
            else:
                print("[gpu] no CUDA device, running YOLO on CPU")
        except Exception:
            DEVICE = "cpu"
            print("[gpu] torch not importable, running YOLO on CPU")
    return DEVICE


def load_yolo(path):
    from ultralytics import YOLO
    m = YOLO(path)
    dev = pick_device()
    try:
        m.to(dev)
    except Exception:
        pass
    return m


def infer_masks(model, frame):
    """ONE YOLO inference -> (floor_mask, wall_mask).

    Post-processing is copied verbatim from compute_floor_mask and
    wall_suspicion_mask; both thresholds are 0.20, so the raw union is
    shared and only the morphology differs. This is the fusion that
    halves the model workload.
    """
    h, w = frame.shape[:2]
    res = model(frame, verbose=False, device=DEVICE)[0]
    wall_f = np.zeros((h, w), dtype=np.uint8)      # for the floor mask
    wall_s = np.zeros((h, w), dtype=np.uint8)      # for the seg mask
    if res.masks is not None and res.boxes is not None:
        conf = res.boxes.conf.cpu().numpy()
        keep_f = conf >= WALL_SUSPICION_CONF
        keep_s = conf >= SEG_CONF
        if keep_f.any():
            m = (res.masks.data.cpu().numpy()[keep_f].sum(axis=0) > 0)
            wall_f = cv2.resize(m.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST)
        if keep_s.any():
            m = (res.masks.data.cpu().numpy()[keep_s].sum(axis=0) > 0)
            wall_s = cv2.resize(m.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST)
    if WALL_DILATE_PX > 0:
        wall_f = cv2.dilate(wall_f, np.ones((WALL_DILATE_PX,) * 2, np.uint8))
    floor_mask = ((wall_f == 0) * 255).astype(np.uint8)
    if MASK_OPEN > 1:
        wall_s = cv2.morphologyEx(wall_s, cv2.MORPH_OPEN,
                                  np.ones((MASK_OPEN, MASK_OPEN), np.uint8))
    return floor_mask, wall_s


def compute_floor_mask(model, frame):
    """255 where features are ALLOWED (floor), 0 where banned
    (suspected wall + dilated safety margin)."""
    h, w = frame.shape[:2]
    res = model(frame, verbose=False)[0]
    wall = np.zeros((h, w), dtype=np.uint8)
    if res.masks is not None and res.boxes is not None:
        keep = res.boxes.conf.cpu().numpy() >= WALL_SUSPICION_CONF
        if keep.any():
            m = (res.masks.data.cpu().numpy()[keep].sum(axis=0) > 0)
            wall = cv2.resize(m.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST)
    if WALL_DILATE_PX > 0:
        wall = cv2.dilate(wall, np.ones((WALL_DILATE_PX,) * 2, np.uint8))
    return ((wall == 0) * 255).astype(np.uint8)


def wall_suspicion_mask(model, frame):
    h, w = frame.shape[:2]
    res = model(frame, verbose=False)[0]
    if res.masks is None or res.boxes is None:
        return np.zeros((h, w), dtype=np.uint8)
    keep = res.boxes.conf.cpu().numpy() >= SEG_CONF
    if not keep.any():
        return np.zeros((h, w), dtype=np.uint8)
    m = (res.masks.data.cpu().numpy()[keep].sum(axis=0) > 0).astype(np.uint8)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    if MASK_OPEN > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             np.ones((MASK_OPEN, MASK_OPEN), np.uint8))
    return m


# ============================= RAY ENGINE ====================================

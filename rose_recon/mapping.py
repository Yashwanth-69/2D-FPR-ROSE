"""STAGE 2 -- wall grid from frames.

Stamps the per-frame wall masks into one global occupancy grid using the pose
stage 1 recovered, then cleans it. Writes wall_raw.npy, the raw accumulation the
spectral filter then works on.

WallMapper is the incremental form: stage 1 hands each frame straight to it so a
single video pass feeds both stages.

Lifted unchanged from the SAR flyover pipeline this project was extracted from.
"""

import csv
import json
import math
import os
import sys

import cv2
import numpy as np

from . import config
from .config import *  # noqa: F401,F403
from .perception import load_yolo, wall_suspicion_mask

"""
Wall mapper v22 — v21 + TRAJECTORY-GATED GAP EXTENSION (occlusion bridging).

The user's rule, implemented literally:
  - A ray's first floor->wall transition stamps that cell as WALL,
    immediately and permanently. No weights, no log-odds, no minimum hit
    counts, no thresholds. The live map shows it the same frame.
  - Free space is tracked only for visualization/diagnostics; it can NEVER
    erase a wall cell.
  - ALL noise handling happens in post-processing, structurally:
    connect dotted hits (small close) -> keep components that are LINE-LIKE
    (long & thin) or CONNECTED to other kept walls -> drop confetti ->
    optional Hough + dominant-axis snap -> 4 px walls -> centered 600x600.
  - ANNULUS SNAP (v18 fix): snap only applies to hits landing at
    SNAP_MIN_CELLS..SNAP_CELLS from an existing wall (genuine drift
    duplicates). Hits CLOSER than SNAP_MIN_CELLS stamp themselves — in v17
    the full-disc snap teleported gap-filling hits onto existing dots, so
    the first frame's dot pattern fossilized into permanent dashed lines.
    The annulus lets gaps along a wall fill while still merging duplicates.

Keys (SHOW_LIVE): q quit(+save) | p pause | s snapshot | f follow/overview
Tuning layers in mapping_data/tuning/.

SUBMAPS (new in v19, from GPS-denied mapping research): rays stamp into a
small SUBMAP covering ~SUBMAP_ROWS pose rows — short enough that odometry
drift within it stays below one map cell, so each submap is internally
consistent. On completion the submap is aligned to the global map by a
translation-only image correlation (heading is compass-absolute): slide the
submap's wall image over the global wall image within +-SEARCH_CELLS and
maximize wall-on-wall overlap (global dilated 1 cell for tolerance). The
winning offset composites the submap in and becomes the prior for the next
submap, so the correction TRACKS drift. A submap with too little wall
content, or too little overlap with the global map, pastes at the carried
offset — no wild guesses. Re-seen walls slide onto their originals: one
wall in the output. This replaces per-frame scan matching (too sparse) and
feature loop closure (repetitive floor texture) with whole-structure
alignment at the right granularity.
"""

# ============================= CONFIG ========================================

# SEG_Z0 SUBTRACTION LIKELY A BUG: aruco altitude z = f*375/d is ALREADY the
# height above the mat (= floor). Subtracting SEG_Z0 again shortens every
# projected range by ~4% (96/2400), pulling walls toward the drone from
# every viewing side -> ~25 cm doubling at 3 m range, while odometry travel
# (which never subtracts SEG_Z0) stays correct. Set 0.0 to test; restore 96.48
# to reproduce old behavior.
SEG_Z0           = 0.0
FOCAL_LENGTH = 569.84

N_RAYS       = 900        # angular resolution (0.4 deg)
RAY_STEP_PX  = 3          # radial march step

MAP_RES_M    = 0.05
WORK_PX      = 1200

# --- Snap-to-wall (loop-closure duplicates) ---
SNAP_ON      = True
SNAP_CELLS   = 6          # snap hits within 0.30 m of existing wall onto it
SNAP_MIN_CELLS = 3        # ...but only if at least this far away: closer
                          # hits stamp themselves (fill gaps along the wall)

# --- Submap architecture ---
SUBMAP_ROWS   = 350       # pose rows per submap (drift within < 1 cell)
SEARCH_CELLS  = 12        # +-0.6 m alignment search around carried offset
MIN_SUB_WALL  = 150       # submap wall cells needed to attempt alignment
MIN_OVERLAP   = 60        # aligned overlap cells needed to accept an offset
STEP_CAP      = 6         # max offset change between consecutive submaps
SNAP_REFRESH = 20         # rebuild nearest-wall lookup every N frames


# --- Contradiction filter (non-temporal noise removal) ---
# A component is noise if the free layer CONTRADICTS it: rays from other
# frames passed through its cells seeing floor. Real walls — even ones
# stamped in a single frame — are never floor-crossed (rays hit them or are
# occluded; the carve margin keeps near-miss floor votes off the band).
# Intensity/hit-counts are never used, so once-seen walls survive.
                          # cells were also carved as observed floor
                          # (1-2 px flecks never generate hits at all)

# --- Rectilinear pruning (furniture-dent removal) ---
# Furniture hugging a wall occludes the base; rays hit the wall's UPPER
# part above it, which projects overshot -> a bump on the straight line.
# Fix: directional opening in the dominant-axis frame — a cell survives
# only if it belongs to a straight run >= RECT_LEN cells along the
# horizontal OR vertical axis. Long walls trivially qualify; furniture-
# shaped bumps (short both ways) are shaved flush. Prune-only: never adds.
RECT_LEN       = 9        # 0.45 m minimum straight run to keep a cell

# --- Trajectory-gated gap extension (occlusion bridging) ---
# Furniture hugging walls occludes the base -> gaps in straight wall runs.
# Fix: extend each fitted wall segment from its endpoints along its own
# direction; FILL the run only if it terminates on another wall within
# MAX_EXTEND_CELLS. ABORT (leave the gap) if the march touches the drone's
# flight corridor — the drone flew through every doorway, so doorways are
# certified not-wall and can never be sealed; furniture gaps are places the
# drone could not fly. The user's constraint, verbatim.
# --- DRONE FOOTPRINT CLEARING (physical presence = certified free) ---
# A wall cannot exist where the drone itself flew. Every frame, a square of
# half-width FOOT_CELLS around the drone's grid cell is cleared of wall
# stamps and LOCKED as free (noise can never restamp it). Tunable live via
# the 'Footprint' trackbar; the final value also sweeps the whole
# trajectory at the end, so earlier frames get the same treatment.
FOOT_CELLS     = 5     # half-width in cells (5 = 0.55 m square)
FOOT_MAX       = 20    # trackbar range

MAX_EXTEND_CELLS = 50     # bridge at most 2.5 m

SHOW_LIVE    = True
VIS_CAM_W    = 860
VIS_VIEW_PX  = 520
VIS_ZOOM     = 1.3

# ============================= GEOMETRY ======================================
def project_ground(us, vs, pose, w, h):
    x_cam, y_cam, z_cam, theta = pose
    H_mm = z_cam - SEG_Z0
    if H_mm <= 10:
        return None, None
    uc, vc = w / 2.0, h / 2.0
    x_loc = ((us - uc) * H_mm) / FOCAL_LENGTH
    y_loc = -((vs - vc) * H_mm) / FOCAL_LENGTH
    c, s = np.cos(theta), np.sin(theta)
    xg = x_cam + (x_loc * c + y_loc * s)
    yg = y_cam + (-x_loc * s + y_loc * c)
    return xg / 1000.0, yg / 1000.0


def world_to_grid(x_m, y_m):
    col = np.round(np.asarray(x_m) / MAP_RES_M + WORK_PX / 2.0).astype(np.int64)
    row = np.round(WORK_PX / 2.0 - np.asarray(y_m) / MAP_RES_M).astype(np.int64)
    return row, col


# ============================= PERCEPTION ====================================
def ray_observations(mask):
    """First floor->wall transition per ray = hit. Free samples = everything
    crossed before the hit. No weights — a hit is a hit."""
    h, w = mask.shape
    uc, vc = w / 2.0, h / 2.0
    max_r = np.hypot(uc, vc)

    angles = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    radii = np.arange(0, max_r, RAY_STEP_PX)

    us = uc + radii[None, :] * np.cos(angles)[:, None]
    vs = vc + radii[None, :] * np.sin(angles)[:, None]
    inside = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
    ui = np.clip(us.astype(np.int32), 0, w - 1)
    vi = np.clip(vs.astype(np.int32), 0, h - 1)
    samples = mask[vi, ui].astype(np.int8)
    samples[~inside] = -1

    free_u, free_v = [], []
    hit_u, hit_v = [], []

    for ray in range(N_RAYS):
        s = samples[ray]
        exited = s[0] != 1
        for k in range(len(s)):
            if s[k] == -1:
                break
            if not exited:
                if s[k] == 0:
                    exited = True
                continue
            if s[k] == 0:
                free_u.append(us[ray, k])
                free_v.append(vs[ray, k])
            else:
                hit_u.append(us[ray, k])
                hit_v.append(vs[ray, k])
                break  # occluded beyond
    return (np.array(free_u), np.array(free_v),
            np.array(hit_u), np.array(hit_v))


# ============================= SNAP LOOKUP ===================================
def build_snap_lookup(wall):
    if not wall.any():
        return None
    src_img = np.where(wall, 0, 255).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(
        src_img, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    coords = np.argwhere(wall)   # row-major, matches DIST_LABEL_PIXEL order
    return dist, labels, coords


# ============================= SUBMAP ALIGNMENT ==============================
def align_submap(global_wall, sub_wall, prior_dr, prior_dc):
    """Translation-only correlation of the submap wall image against the
    (1-cell dilated) global wall image, searched around the carried prior.
    Returns (dr, dc, accepted)."""
    ys, xs = np.nonzero(sub_wall)
    if ys.size < MIN_SUB_WALL or not global_wall.any():
        return prior_dr, prior_dc, False
    gdil = cv2.dilate(global_wall.astype(np.uint8),
                      np.ones((3, 3), np.uint8)).astype(bool)
    best_score, best = -1, (prior_dr, prior_dc)
    for dr in range(prior_dr - SEARCH_CELLS, prior_dr + SEARCH_CELLS + 1):
        r = ys + dr
        rok = (r >= 0) & (r < WORK_PX)
        for dc in range(prior_dc - SEARCH_CELLS, prior_dc + SEARCH_CELLS + 1):
            c = xs + dc
            ok = rok & (c >= 0) & (c < WORK_PX)
            if not ok.any():
                continue
            score = int(gdil[r[ok], c[ok]].sum())
            if dr == prior_dr and dc == prior_dc:
                score = int(score * 1.03)      # stay-put tie break
            if score > best_score:
                best_score, best = score, (dr, dc)
    if best_score < MIN_OVERLAP:
        return prior_dr, prior_dc, False
    dr = int(np.clip(best[0], prior_dr - STEP_CAP, prior_dr + STEP_CAP))
    dc = int(np.clip(best[1], prior_dc - STEP_CAP, prior_dc + STEP_CAP))
    return dr, dc, True


def composite(global_wall, global_free, sub_wall, sub_free, dr, dc):
    ys, xs = np.nonzero(sub_wall)
    r, c = ys + dr, xs + dc
    ok = (r >= 0) & (r < WORK_PX) & (c >= 0) & (c < WORK_PX)
    global_wall[r[ok], c[ok]] = True
    ys, xs = np.nonzero(sub_free)
    r, c = ys + dr, xs + dc
    ok = (r >= 0) & (r < WORK_PX) & (c >= 0) & (c < WORK_PX)
    global_free[r[ok], c[ok]] = True


# ============================= ACCUMULATE ====================================
def accumulate(wall, seen_free, mask, pose, w, h, snap=None):
    fu, fv, hu, hv = ray_observations(mask)

    if hu.size:
        xg, yg = project_ground(hu, hv, pose, w, h)
        if xg is not None:
            r, c = world_to_grid(xg, yg)
            ok = (r >= 0) & (r < WORK_PX) & (c >= 0) & (c < WORK_PX)
            r, c = r[ok], c[ok]
            if SNAP_ON and snap is not None:
                dist, labels, coords = snap
                d = dist[r, c]
                near = (d >= SNAP_MIN_CELLS) & (d <= SNAP_CELLS)
                if near.any():
                    lbl = labels[r[near], c[near]] - 1
                    tgt = coords[lbl]
                    r[near] = tgt[:, 0]
                    c[near] = tgt[:, 1]
            wall[r, c] = True          # DIRECT, PERMANENT

    if fu.size:
        xg, yg = project_ground(fu, fv, pose, w, h)
        if xg is not None:
            r, c = world_to_grid(xg, yg)
            ok = (r >= 0) & (r < WORK_PX) & (c >= 0) & (c < WORK_PX)
            seen_free[r[ok], c[ok]] = True   # visualization only
    return True


# ============================= POST-PROCESSING ===============================




# ============================= VISUALIZATION =================================
def draw_camera_view(frame, mask):
    vis = frame.copy()
    overlay = vis.copy()
    overlay[mask == 0] = (0, 200, 0)
    overlay[mask > 0] = (255, 80, 0)
    vis = cv2.addWeighted(overlay, 0.30, vis, 0.70, 0)
    h, w = vis.shape[:2]
    cv2.drawMarker(vis, (w // 2, h // 2), (0, 255, 255),
                   cv2.MARKER_CROSS, 30, 2)
    return cv2.resize(vis, (VIS_CAM_W, int(h * VIS_CAM_W / w)))


def render_map(wall, seen_free, traj_rc, drone_rc, yaw,
               frame_idx, total, follow=True):
    img = np.full((WORK_PX, WORK_PX, 3), 30, dtype=np.uint8)
    img[seen_free & ~wall] = (60, 130, 60)
    img[wall] = (0, 60, 255)               # every stamp visible immediately

    cx = cy = WORK_PX // 2
    cv2.line(img, (cx, cy), (cx + 20, cy), (0, 0, 200), 1)
    cv2.line(img, (cx, cy), (cx, cy - 20), (0, 200, 0), 1)
    cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1)
    for k in range(1, len(traj_rc)):
        cv2.line(img, (traj_rc[k - 1][1], traj_rc[k - 1][0]),
                 (traj_rc[k][1], traj_rc[k][0]), (200, 120, 0), 1)
    if drone_rc is not None:
        r, c = drone_rc
        if 0 <= r < WORK_PX and 0 <= c < WORK_PX:
            cv2.circle(img, (c, r), 5, (0, 255, 0), -1)
            cv2.arrowedLine(img, (c, r),
                            (int(c + 18 * np.sin(yaw)),
                             int(r - 18 * np.cos(yaw))),
                            (255, 255, 255), 2, tipLength=0.35)

    if follow and drone_rc is not None:
        half = VIS_VIEW_PX // 2
        r0 = int(np.clip(drone_rc[0] - half, 0, WORK_PX - VIS_VIEW_PX))
        c0 = int(np.clip(drone_rc[1] - half, 0, WORK_PX - VIS_VIEW_PX))
        view = img[r0:r0 + VIS_VIEW_PX, c0:c0 + VIS_VIEW_PX]
        view = cv2.resize(view, None, fx=VIS_ZOOM, fy=VIS_ZOOM,
                          interpolation=cv2.INTER_NEAREST)
        mode = "FOLLOW"
    else:
        view = cv2.resize(img, (int(VIS_VIEW_PX * VIS_ZOOM),) * 2,
                          interpolation=cv2.INTER_AREA)
        mode = "OVERVIEW"

    hud = [f"{mode}  frame {frame_idx}/{total}",
           f"wall px: {int(wall.sum())}   free px: "
           f"{int(seen_free.sum())}",
           "q quit | p pause | s snapshot | f follow/overview"]
    for j, line in enumerate(hud):
        cv2.putText(view, line, (10, 22 + 20 * j),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return view


# ============================= MAIN ==========================================
class WallMapper:
    """Streaming form of run_mapping: identical state machine, but fed one
    (frame, pose, wall_mask) at a time so it can run inside the odometry
    loop. feed() is called once per SAVED pose row, in order; the row
    counter drives FRAME_STRIDE exactly like the legacy CSV loop."""

    def __init__(self, total_hint=0):
        os.makedirs(os.path.dirname(config.OUT_PATH), exist_ok=True)
        os.makedirs(config.TUNING_DIR, exist_ok=True)
        self.global_wall = np.zeros((WORK_PX, WORK_PX), dtype=bool)
        self.global_free = np.zeros((WORK_PX, WORK_PX), dtype=bool)
        self.sub_wall = np.zeros((WORK_PX, WORK_PX), dtype=bool)
        self.sub_free = np.zeros((WORK_PX, WORK_PX), dtype=bool)
        self.sub_rows = 0
        self.off_r, self.off_c = 0, 0
        self.n_aligned = 0
        self.traj_rc = []
        self.n_frames = 0
        self.snap = None
        self.paused, self.follow = False, True
        self.foot_lock = np.zeros((WORK_PX, WORK_PX), dtype=bool)
        self.row_idx = 0
        self.total = total_hint
        self.quit = False

    def _clear_foot(self, grid_wall, grid_free, rc, half):
        r0 = max(rc[0] - half, 0); r1 = min(rc[0] + half + 1, WORK_PX)
        c0 = max(rc[1] - half, 0); c1 = min(rc[1] + half + 1, WORK_PX)
        grid_wall[r0:r1, c0:c1] = False
        grid_free[r0:r1, c0:c1] = True
        self.foot_lock[r0:r1, c0:c1] = True

    def _finish_submap(self):
        if self.sub_rows == 0:
            return
        ndr, ndc, ok = align_submap(self.global_wall, self.sub_wall,
                                    self.off_r, self.off_c)
        if ok:
            self.n_aligned += 1
            print(f"[submap] aligned: offset ({ndr},{ndc}) cells "
                  f"= ({ndc*MAP_RES_M*1000:+.0f},"
                  f"{-ndr*MAP_RES_M*1000:+.0f}) mm")
        else:
            print(f"[submap] pasted at carried offset "
                  f"({self.off_r},{self.off_c}) "
                  f"(insufficient wall content or overlap)")
        self.off_r, self.off_c = ndr, ndc
        composite(self.global_wall, self.global_free,
                  self.sub_wall, self.sub_free, self.off_r, self.off_c)
        self.global_wall[self.foot_lock] = False
        self.global_free[self.foot_lock] = True
        self.sub_wall = np.zeros((WORK_PX, WORK_PX), dtype=bool)
        self.sub_free = np.zeros((WORK_PX, WORK_PX), dtype=bool)
        self.sub_rows = 0

    def feed(self, frame, pose_row, mask):
        """One saved pose row. mask may be None on strided-out rows
        (it is not used then), matching the legacy loop which skipped
        those rows entirely."""
        i = self.row_idx
        self.row_idx += 1
        if i % FRAME_STRIDE:
            return
        if frame is None:
            return
        h, w = frame.shape[:2]
        pose = (pose_row["global_x_mm"], pose_row["global_y_mm"],
                pose_row["altitude_z_mm"], pose_row["yaw_rad"])

        if SNAP_ON and self.n_frames % SNAP_REFRESH == 0:
            self.snap = build_snap_lookup(self.sub_wall)
        if accumulate(self.sub_wall, self.sub_free, mask, pose, w, h,
                      snap=self.snap):
            self.n_frames += 1
            self.sub_rows += 1
            if self.sub_rows >= SUBMAP_ROWS:
                self._finish_submap()

        dr, dc = world_to_grid(pose_row["global_x_mm"] / 1000.0,
                               pose_row["global_y_mm"] / 1000.0)
        drone_rc = (int(dr), int(dc))
        self.traj_rc.append(drone_rc)
        try:
            FOOT = cv2.getTrackbarPos("Footprint", "Live Map (submaps)")
        except cv2.error:
            FOOT = FOOT_CELLS
        self._clear_foot(self.sub_wall, self.sub_free, drone_rc, FOOT)
        self.global_wall[self.foot_lock] = False
        self.global_free[self.foot_lock] = True

        if SHOW_LIVE and self.n_frames == 1:
            cv2.namedWindow("Live Map (submaps)")
            cv2.createTrackbar("Footprint", "Live Map (submaps)",
                               FOOT_CELLS, FOOT_MAX, lambda v: None)
        if SHOW_LIVE:
            view_wall = self.global_wall.copy()
            view_free = self.global_free.copy()
            composite(view_wall, view_free, self.sub_wall, self.sub_free,
                      self.off_r, self.off_c)
            cv2.imshow("Camera (blue=wall, green=free)",
                       draw_camera_view(frame, mask))
            cv2.imshow("Live Map (submaps)",
                       render_map(view_wall, view_free, self.traj_rc,
                                  drone_rc, pose_row["yaw_rad"], i,
                                  max(self.total, i + 1), self.follow))
            key = cv2.waitKey(0 if self.paused else 1) & 0xFF
            if key == ord('q'):
                print("[vis] quit — saving what we have.")
                self.quit = True
            elif key == ord('p'):
                self.paused = not self.paused
            elif key == ord('f'):
                self.follow = not self.follow
            elif key == ord('s'):
                cv2.imwrite(os.path.join(config.TUNING_DIR, "snap_map.png"),
                            render_map(self.global_wall, self.global_free,
                                       self.traj_rc, drone_rc,
                                       pose_row["yaw_rad"], i,
                                       max(self.total, i + 1), self.follow))
                cv2.imwrite(os.path.join(config.TUNING_DIR, "snap_cam.png"),
                            draw_camera_view(frame, mask))
                print("[vis] snapshots saved.")
        elif i % 300 == 0:
            print(f"[map] row {i}  wall px: {int(self.global_wall.sum())}")

    def finish(self):
        self._finish_submap()
        try:
            FOOT = cv2.getTrackbarPos("Footprint", "Live Map (submaps)")
        except cv2.error:
            FOOT = FOOT_CELLS
        for rc in self.traj_rc:
            self._clear_foot(self.global_wall, self.global_free, rc, FOOT)
        print(f"[foot] trajectory sweep done with half-width {FOOT} cells "
              f"({(2*FOOT+1)*MAP_RES_M:.2f} m square)")
        if SHOW_LIVE:
            cv2.destroyWindow("Live Map (submaps)") if False else None
        print(f"[map] frames accumulated: {self.n_frames}, submaps aligned: "
              f"{self.n_aligned}, raw wall cells: "
              f"{int(self.global_wall.sum())}, final carried offset: "
              f"({self.off_r},{self.off_c}) cells")
        if self.n_frames == 0:
            raise RuntimeError("No frames accumulated — check paths/CSV units.")
        np.save(config.WALL_NPY, self.global_wall)
        np.save(os.path.join(config.MAPPING_DIR, "seen_free.npy"),
                self.global_free)
        # The inherited stage-2 tail vectorised this grid into a single-line
        # footprint and wrote r0_footprint_estimate.png. Removed: this project
        # stops at the spectral filter, nothing read that image, and on a map
        # with no straight Hough lines it printed "[final] WARNING: blank map",
        # which read like a failed run when the run was fine. wall_raw.npy is
        # already saved above and is unaffected.


def run_mapping():
    """Legacy STAGE 2: read saved frames + CSV from disk. Only needed when
    SAVE_FRAMES was on and you want to re-map without re-flying the video."""
    import pandas as pd
    os.makedirs(os.path.dirname(config.OUT_PATH), exist_ok=True)
    os.makedirs(config.TUNING_DIR, exist_ok=True)
    print("Loading model and poses...")
    model = load_yolo(config.MODEL_PATH)
    df = pd.read_csv(config.CSV_PATH)
    print(f"[sanity] altitude above floor: "
          f"min={df['altitude_z_mm'].min() - SEG_Z0:.0f} mm, "
          f"max={df['altitude_z_mm'].max() - SEG_Z0:.0f} mm")
    span_x = (df['global_x_mm'].max() - df['global_x_mm'].min()) / 1000.0
    span_y = (df['global_y_mm'].max() - df['global_y_mm'].min()) / 1000.0
    print(f"[sanity] trajectory span: {span_x:.1f} x {span_y:.1f} m")

    mapper = WallMapper(total_hint=len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        # mask only computed for rows the mapper will use (stride)
        mask = None
        frame = None
        if i % FRAME_STRIDE == 0:
            frame = cv2.imread(os.path.join(config.FRAMES_DIR, row["frame_file"]))
            if frame is not None:
                mask = wall_suspicion_mask(model, frame)
        mapper.feed(frame, row, mask)
        if mapper.quit:
            break
    if SHOW_LIVE:
        cv2.destroyAllWindows()
    mapper.finish()




# ============================= VISUALIZATION =================================




# ============================= MAIN ==========================================









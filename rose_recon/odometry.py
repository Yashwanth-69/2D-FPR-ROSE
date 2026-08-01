"""STAGE 1 -- visual odometry.

Recovers the flight path from the footage. Scale and the reference frame come
from the origin-marker ArUco patterns; altitude follows from the known marker
size. Writes camera_poses.csv.

Lifted unchanged from the SAR flyover pipeline this project was extracted from,
so the trajectory it produces is identical.
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
from .perception import (pick_device, load_yolo, infer_masks,
                         compute_floor_mask, wall_suspicion_mask)

# --- MATHEMATICALLY VERIFIED CAMERA PARAMETERS (Phase 1) ---
Z0 = 96.48            # True Initial Altitude (mm)
FOCAL_LENGTH = 569.84 # True Focal Length (pixels)
FPS = 62.5
MARKER_DISTANCE_MM = 375.0 # True physical distance between ArUco centers
MARKER_DIAG_DISTANCE_MM = 375.0 * np.sqrt(2)  # pair (1,2) is the diagonal

# --- GYRO COMPENSATION CONFIGURATION ---
GYRO_ROLL_SIGN = 1.0  
GYRO_PITCH_SIGN = 1.0 

# --- KINEMATIC GATING (The Experiment) ---
# Maximum allowed discrepancy between IMU expected distance and Camera distance per frame.
# Increased to 40mm to reduce the "parachute" dampening effect while still blocking teleports.
MAX_DEVIATION_MM = 40.0 

# --- FLOOR-ONLY FEATURE TRACKING (the one change vs the original) ---
# Wall/object features at height h produce flow amplified by z/(z-h) vs
# floor features; the median gets captured by them in wall-heavy frames,
# biasing translation. Spawn features on floor only; evict wall-wanderers.
USE_FLOOR_MASK   = True
WALL_SUSPICION_CONF = 0.20   # loose: anything suspected wall is banned
WALL_DILATE_PX   = 10        # safety margin around wall masks (px)
MASK_EVERY_N     = 2         # recompute floor mask every N frames
MIN_SPAWN_FLOOR  = 25        # fewer masked corners than this -> spawn unmasked
MIN_KEEP_POINTS  = 15        # eviction never drops the pool below this

# --- Optical Flow Parameters ---
feature_params = dict(maxCorners=250, qualityLevel=0.07, minDistance=15, blockSize=7)
lk_params = dict(winSize=(21, 21), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# --- Z SOURCE (consistency fix, learned from final_1 comparison) ---
# The foveated visual scale is the noisiest instrument in the pipeline and
# continuously modulates current_z = the mm-per-pixel ruler. A corridor
# traversed at believed z=2.40 and retraced at z=2.31 yields legs of
# different lengths -> retrace misalignment. The flights hold altitude in
# cruise, so: z comes from the ArUco anchor (absolute, ID-correct) plus the
# IMU vertical channel for genuine climbs. Visual z is disabled.
Z_FROM_VISION = False

# --- Navigation Constraints ---
ROTATION_THRESHOLD = 0.003 # Radians per frame

def load_imu_data(csv_path):
    import pandas as pd
    """Calculates Absolute Heading, Pure Linear Accel (X,Y,Z), and Raw Gyro Rates."""
    df = pd.read_csv(csv_path)
    t = df['timestamp'].values
    if np.max(t) > 1000: t = t / 1000.0
    
    dt = np.diff(t)
    dt = np.insert(dt, 0, 0)
    
    # 1. Compass Heading
    comp_x = df['comp_x'].values
    comp_y = df['comp_y'].values
    yaw_rad = np.arctan2(-comp_x, comp_y)
    
    # 2. Gyroscope Rates (Raw instantaneous, no drifting integration!)
    w_x = df['w_x'].values
    w_y = df['w_y'].values
    
    # 3. Accelerometer processing (Remove gravity and tilt leakage)
    roll = np.cumsum(w_x * dt)
    pitch = np.cumsum(w_y * dt)
    
    ax = df['acc_x'].values
    ay = df['acc_y'].values
    az = df['acc_z'].values
    
    # Gravity removal in local frame based on tilt
    g = 9.81
    g_x = -g * np.sin(pitch)
    g_y = g * np.sin(roll) * np.cos(pitch)
    g_z = g * np.cos(roll) * np.cos(pitch)
    
    ax_linear = ax - g_x
    ay_linear = ay - g_y
    az_linear = az - g_z
    
    return t, yaw_rad, ax_linear, ay_linear, az_linear, w_x, w_y

def aruco_altitude(corners, ids):
    """Absolute altitude from a SPECIFIC ArUco marker pair, selected by ID.
    Pairs (0,1)/(0,2) are 375 mm apart; (1,2) is the 375*sqrt(2) diagonal.
    (The original code used the first two DETECTED markers, which silently
    inflated altitude by sqrt(2) whenever the detector returned pair 1&2.)"""
    centers = {}
    for c, mid in zip(corners, ids.flatten()):
        centers[int(mid)] = np.mean(c[0], axis=0)

    for a, b, dist_mm in ((0, 1, MARKER_DISTANCE_MM),
                          (0, 2, MARKER_DISTANCE_MM),
                          (1, 2, MARKER_DIAG_DISTANCE_MM)):
        if a in centers and b in centers:
            pixel_distance = np.linalg.norm(centers[a] - centers[b])
            if pixel_distance > 5:
                return FOCAL_LENGTH * (dist_mm / pixel_distance)
    return None

def image_to_global(u, v, drone_pos, theta, orig_w, orig_h):
    """Project a pixel (u, v) of the down-facing flyover camera to global map
    coordinates in millimetres. This is the exact projection tested in
    victims_location.py (pinhole inversion with the Z0 altitude fix)."""
    x_cam, y_cam, z_cam_mm = drone_pos
    h_mm = z_cam_mm                       # altitude already = height above floor
    if h_mm <= 10:
        return None
    uc, vc = orig_w / 2.0, orig_h / 2.0
    x_local = ((u - uc) * h_mm) / FOCAL_LENGTH
    y_local = -((v - vc) * h_mm) / FOCAL_LENGTH
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_global = x_cam + (x_local * cos_t + y_local * sin_t)
    y_global = y_cam + (-x_local * sin_t + y_local * cos_t)
    return (x_global, y_global)


def clamp_f(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _weighted_geometric_median(points, weights, iters=64):
    """Outlier-robust 2D centre (Weiszfeld). Fallback + sanity anchor."""
    P = np.average(points, axis=0, weights=weights)
    for _ in range(iters):
        d = np.maximum(np.linalg.norm(points - P, axis=1), 1e-6)
        P_new = np.average(points, axis=0, weights=weights / d)
        if np.linalg.norm(P_new - P) < 1e-7:
            break
        P = P_new
    return P


def _parallax_stats(members):
    """Weighted mean and covariance of the per-view parallax vector
        p_i = (ground projection - camera xy) / altitude = tan(beta) * azimuth.

    Returns (mu, spread_max, spread_min, major_dir). `mu` is the mean parallax:
    the ground projections are displaced from the true position by
    body_height * mu, so mu points ALONG the bias (from the true victim toward
    where the projections land). `spread_max` is the largest variance of p over
    the cluster and is the observability of the target's height: no variation
    means the bias is a constant that cannot be estimated from the data."""
    P, W = [], []
    for m in members:
        C = m["C"]
        H = max(0.1, C[2])
        P.append([(m["x_m"] - C[0]) / H, (m["y_m"] - C[1]) / H])
        W.append(max(1e-6, m["w"]))
    P = np.asarray(P, float)
    W = np.asarray(W, float)
    W = W / W.sum()
    mu = (P * W[:, None]).sum(0)
    D = P - mu
    S = (D * W[:, None]).T @ D
    evals, evecs = np.linalg.eigh(S)
    return (mu, float(max(evals[1], 0.0)), float(max(evals[0], 0.0)),
            evecs[:, 1])


def _triangulate_rays(members):
    """Huber-robust least-squares closest point to a cluster's viewing rays. Each
    detection gives a ray (camera centre C_i, unit dir d_i); P minimises the
    weighted sum of squared perpendicular distances, A P = b with
    A = sum w_i (I - d_i d_i^T). IRLS rejects outliers. Returns (x, y) or None on
    a singular solve / implausible recovered height. Whether this result should
    be TRUSTED is decided by _parallax_stats, not here: the solve can be
    numerically fine and still be physically unobservable."""
    Cs, ds, w0 = [], [], []
    for m in members:
        C = np.asarray(m["C"], float)
        v = np.array([m["x_m"], m["y_m"], 0.0], float) - C
        n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        Cs.append(C); ds.append(v / n); w0.append(max(1e-3, m["w"]))
    if len(Cs) < 3:
        return None
    Cs, ds, w0 = np.asarray(Cs), np.asarray(ds), np.asarray(w0)
    P = None
    for _ in range(IRLS_ITERS):
        A = np.zeros((3, 3)); b = np.zeros(3)
        for i in range(len(Cs)):
            M = np.eye(3) - np.outer(ds[i], ds[i])
            wi = w0[i]
            if P is not None:
                r = np.linalg.norm(M @ (P - Cs[i]))
                if r > HUBER_DELTA_M:
                    wi *= HUBER_DELTA_M / r
            A += wi * M; b += wi * (M @ Cs[i])
        try:
            P = np.linalg.solve(A + 1e-6 * np.eye(3), b)
        except np.linalg.LinAlgError:
            return None
    if P is None or abs(P[2]) > TRI_MAX_Z_M:
        return None
    return (float(P[0]), float(P[1]))


def _nadir_subset(members):
    """The most-NADIR detections of a cluster: those whose viewing ray is closest
    to straight down. Their ground projections carry the smallest h*tan(beta)
    parallax bias, so when the geometry is degenerate these are the only
    projections worth averaging. Falls back to everything on a small cluster."""
    scored = []
    for m in members:
        C = m["C"]
        d_h = math.hypot(m["x_m"] - C[0], m["y_m"] - C[1])
        H = max(0.1, C[2])
        scored.append((d_h / H, m))             # tan(beta): smaller = more nadir
    scored.sort(key=lambda t: t[0])
    k = max(NADIR_KEEP_MIN, int(round(len(scored) * NADIR_KEEP_FRAC)))
    k = min(k, len(scored))
    return [m for _, m in scored[:k]]


def _robust_victim_location(members):
    """Best (x, y) for a cluster, plus an honest uncertainty for it.

    Enough parallax variation (the drone's viewing geometry genuinely changed
    across the cluster) -> the target's height is observable, so multi-view
    triangulation recovers the true 3D point and the z=0 parallax bias cancels.

    Not enough -> height is unobservable and EVERY view carries the same
    h*tan(beta) offset. Averaging or median-ing the projections cannot remove a
    constant. But that constant is predictable, so we estimate it: take the
    least-biased (most-nadir) views and subtract ASSUMED_BODY_H_M * mean_parallax
    analytically. The leftover uncertainty is the body-height uncertainty
    projected along that same axis, which is exported so the ground robot knows
    which way to sweep.

    Returns (x, y, sigma_major, sigma_minor, weak_ux, weak_uy, quality) with
    quality in {'tri', 'debias', 'median'}."""
    pts = np.array([[m["x_m"], m["y_m"]] for m in members], float)
    w = np.array([m["w"] for m in members], float)
    if w.sum() <= 0:
        w = np.ones(len(members))
    med = _weighted_geometric_median(pts, w)

    mu, spread_max, _spread_min, _major = _parallax_stats(members)
    mu_n = float(np.linalg.norm(mu))
    # Unit vector pointing from the biased projections back toward the truth.
    if mu_n > 1e-6:
        back = (-mu[0] / mu_n, -mu[1] / mu_n)
    else:
        back = (1.0, 0.0)

    observable = spread_max >= PARALLAX_MIN_VAR
    tri = _triangulate_rays(members)

    if observable and tri is not None and \
            math.hypot(tri[0] - med[0], tri[1] - med[1]) <= TRI_TRUST_M:
        # Height solved: residual is dominated by detection noise, not geometry.
        sig = float(clamp_f(0.35 / math.sqrt(max(spread_max, 1e-6)) * 0.05,
                            0.15, 1.20))
        return (tri[0], tri[1], sig, sig * 0.7, back[0], back[1], "tri")

    # Degenerate: de-bias the least-biased views analytically.
    sub = _nadir_subset(members)
    sp = np.array([[m["x_m"], m["y_m"]] for m in sub], float)
    sw = np.array([m["w"] for m in sub], float)
    if sw.sum() <= 0:
        sw = np.ones(len(sub))
    nm = _weighted_geometric_median(sp, sw)
    mu_sub, _sm, _sn, _mj = _parallax_stats(sub)
    x = float(nm[0] - ASSUMED_BODY_H_M * mu_sub[0])
    y = float(nm[1] - ASSUMED_BODY_H_M * mu_sub[1])
    # What is left is our uncertainty in the body height, times the parallax.
    mu_sub_n = float(np.linalg.norm(mu_sub))
    sig_major = float(clamp_f(BODY_H_SIGMA_M * mu_sub_n, 0.25, 2.50))
    if mu_sub_n > 1e-6:
        back = (-mu_sub[0] / mu_sub_n, -mu_sub[1] / mu_sub_n)
    quality = "debias" if tri is not None else "median"
    return (x, y, sig_major, 0.25, back[0], back[1], quality)


def to_origin_marker_frame(x, y):
    """Rotate a point from the pipeline's native frame into the ORIGIN-MARKER
    reference frame the deliverables must use. The offline drone odometry zeroed
    its yaw to compass north rather than the origin marker's +X axis, so the whole
    pipeline map comes out rotated 90 degrees; (x, y) -> (y, -x) rotates it back.

    This MUST be applied before writing victim_location_estimates.csv: the marking
    supervisor adds ONLY the origin translation (coordinate_offset, no rotation)
    before scoring each estimate against the true VICTIM_MARKER world position, so
    an unrotated CSV lands ~90 degrees off in world space and scores ~0 on victim
    localization accuracy (a full half of the 30% Video Information Extraction
    component). The same rotation is applied to wall_estimates.csv so the ground
    controller, which now reads both files without any further rotation, stays in
    one consistent origin-marker frame."""
    return (y, -x)


def project_box_footprint(cx, cy, bx1, by1, bx2, by2, drone_pos, yaw, v_w, v_h):
    """Project the box footprint (centre + 4 corners) to the ground and return the
    centroid in global mm, or None. The centroid follows the body's long axis, so
    for a lying body it lands nearer mid-figure than a single centre pixel."""
    corners = ((cx, cy), (bx1, by1), (bx2, by1), (bx1, by2), (bx2, by2))
    projected = [image_to_global(px, py, drone_pos, yaw, v_w, v_h)
                 for px, py in corners]
    projected = [p for p in projected if p is not None]
    if not projected:
        return None
    return (float(np.mean([p[0] for p in projected])),
            float(np.mean([p[1] for p in projected])))


def finalize_victims(hits):
    """Cluster projected victim hits by spatial proximity, drop clusters seen
    too few times (noise), and write the competition CSV (one 'x,y' per line,
    metres, origin-marker frame, NO header). A richer table with frequency and
    confidence is also saved under src/ for debugging and confidence tuning.
    The contract CSV is always (re)written, empty if nothing was verified."""
    clusters = []
    for h in hits:
        vx, vy, vc = h["x_m"], h["y_m"], h["conf"]
        matched = False
        for c in clusters:
            ax, ay = c["sum_x"] / c["freq"], c["sum_y"] / c["freq"]
            if np.hypot(vx - ax, vy - ay) <= CLUSTER_RADIUS_M:
                c["sum_x"] += vx
                c["sum_y"] += vy
                c["freq"] += 1
                c["max_conf"] = max(c["max_conf"], vc)
                c["members"].append(h)
                matched = True
                break
        if not matched:
            clusters.append(dict(sum_x=vx, sum_y=vy, freq=1, max_conf=vc,
                                 members=[h]))

    # Merge pass: a single victim (especially a large lying one whose ground
    # projections spread out) can seed two separate clusters. Agglomeratively
    # merge clusters whose CENTRES are within VICTIM_MERGE_RADIUS_M, summing
    # frequencies, so one victim yields one estimate. Estimate-count consistency
    # is explicitly part of the 30% Video Info Extraction score, and one victim
    # split in two is the most likely cause of an over-count.
    merged_any = True
    while merged_any and len(clusters) > 1:
        merged_any = False
        for i in range(len(clusters)):
            ci = clusters[i]
            cix, ciy = ci["sum_x"] / ci["freq"], ci["sum_y"] / ci["freq"]
            for j in range(i + 1, len(clusters)):
                cj = clusters[j]
                cjx, cjy = cj["sum_x"] / cj["freq"], cj["sum_y"] / cj["freq"]
                d = float(np.hypot(cix - cjx, ciy - cjy))
                if d <= VICTIM_MERGE_RADIUS_M:
                    print("[victims] MERGE clusters (%.2f,%.2f)+(%.2f,%.2f) "
                          "%.2f m apart, freqs %d+%d -> %d"
                          % (cix, ciy, cjx, cjy, d, ci["freq"], cj["freq"],
                             ci["freq"] + cj["freq"]))
                    ci["sum_x"] += cj["sum_x"]; ci["sum_y"] += cj["sum_y"]
                    ci["freq"] += cj["freq"]
                    ci["max_conf"] = max(ci["max_conf"], cj["max_conf"])
                    ci["members"].extend(cj["members"])
                    clusters.pop(j)
                    merged_any = True
                    break
            if merged_any:
                break

    # Diagnostic: cluster frequency distribution before the noise filter.
    freqs_sorted = sorted((c["freq"] for c in clusters), reverse=True)
    print("[victims] %d clusters after merge, before frequency filter; freqs: %s"
          % (len(clusters), freqs_sorted))

    verified = []
    for c in clusters:
        if c["freq"] < MIN_DETECTION_HITS:
            continue
        rx, ry, smaj, smin, wux, wuy, qual = _robust_victim_location(c["members"])
        mx, my = c["sum_x"] / c["freq"], c["sum_y"] / c["freq"]
        print("[victims]   cluster f=%d: mean=(%.2f,%.2f) -> robust=(%.2f,%.2f) "
              "[%s] parallax correction %.2f m, sigma major %.2f m along "
              "(%+.2f,%+.2f)"
              % (c["freq"], mx, my, rx, ry, qual,
                 np.hypot(rx - mx, ry - my), smaj, wux, wuy))
        if qual in ("debias", "median"):
            print("[victims]     ^ NARROW VIEWING ARC: target height not "
                  "observable, applied h*tan(beta) de-bias; true victim expected "
                  "within ~%.2f m along (%+.2f,%+.2f)" % (smaj, wux, wuy))
        verified.append(dict(x=rx, y=ry, frequency=c["freq"],
                             confidence=c["max_conf"], sigma_major=smaj,
                             sigma_minor=smin, weak_ux=wux, weak_uy=wuy,
                             quality=qual))

    # Optional top-N cap: when the true victim count is known, keep only the N
    # most-seen clusters so we never over-submit (which the scoring penalises).
    if EXPECTED_VICTIM_COUNT is not None and len(verified) > EXPECTED_VICTIM_COUNT:
        verified.sort(key=lambda v: v["frequency"], reverse=True)
        verified = verified[:EXPECTED_VICTIM_COUNT]

    os.makedirs(os.path.dirname(config.VICTIM_OUT_CSV), exist_ok=True)
    with open(config.VICTIM_OUT_CSV, "w") as f:
        for v in verified:
            # Deliverable is scored in the origin-marker frame (see
            # to_origin_marker_frame): bake the rotation in here at source.
            mx, my = to_origin_marker_frame(v["x"], v["y"])
            f.write("%.4f,%.4f\n" % (mx, my))
    # Uncertainty sidecar, row-aligned with the contract CSV above. The contract
    # file must stay a bare x,y list, so the ellipse goes here for the ground
    # controller to read (missing file = controller falls back to a fixed radius).
    os.makedirs(os.path.dirname(config.VICTIM_UNC_CSV), exist_ok=True)
    with open(config.VICTIM_UNC_CSV, "w") as f:
        f.write("sigma_major_m,sigma_minor_m,weak_ux,weak_uy,quality\n")
        for v in verified:
            f.write("%.4f,%.4f,%.4f,%.4f,%s\n"
                    % (v["sigma_major"], v["sigma_minor"],
                       v["weak_ux"], v["weak_uy"], v["quality"]))
    print("[victims] %d raw detection(s), %d cluster(s), %d verified -> %s"
          % (len(hits), len(clusters), len(verified), config.VICTIM_OUT_CSV))
    for i, v in enumerate(verified):
        print("[victims]   victim %d: (%.2f, %.2f) m  seen %d frames, "
              "max conf %.2f, %s, sigma_major %.2f m"
              % (i + 1, v["x"], v["y"], v["frequency"], v["confidence"],
                 v["quality"], v["sigma_major"]))

    if verified:
        dbg = os.path.join(config.MAPPING_DIR, "victim_locations.csv")
        with open(dbg, "w") as f:
            f.write("x_m,y_m,frequency,confidence,sigma_major_m,sigma_minor_m,"
                    "weak_ux,weak_uy,quality\n")
            for v in verified:
                f.write("%.4f,%.4f,%d,%.3f,%.4f,%.4f,%.4f,%.4f,%s\n"
                        % (v["x"], v["y"], v["frequency"], v["confidence"],
                           v["sigma_major"], v["sigma_minor"],
                           v["weak_ux"], v["weak_uy"], v["quality"]))


def run_odometry(video_path, csv_path, mapper=None):
    """STAGE 1, and when mapper is given also STAGE 2, in one video pass.
    One shared YOLO inference per frame feeds BOTH the floor mask (feature
    eviction) and the wall mask (mapping)."""
    import pandas as pd
    import matplotlib.pyplot as plt
    print("Loading IMU Data...")
    imu_t, imu_yaw, imu_ax, imu_ay, imu_az, imu_wx, imu_wy = load_imu_data(csv_path)
    
    # --- Floor-mask model (shared with the mapper when fused) ---
    floor_model = None
    if USE_FLOOR_MASK or mapper is not None:
        try:
            print("Loading wall model (shared floor + wall masks)...")
            floor_model = load_yolo(config.MODEL_PATH)
        except Exception as e:
            print(f"WARNING: floor mask disabled ({e}) — running with "
                  f"unmasked features like the original.")
            if mapper is not None:
                raise SystemExit("fused mapping needs the YOLO model") from e

    # --- Victim detector (runs in this same pass; nothing saved to disk) ---
    victim_model = None
    if DETECT_VICTIMS:
        try:
            print(f"Loading victim detector ({os.path.basename(config.VICTIM_MODEL_PATH)})...")
            victim_model = load_yolo(config.VICTIM_MODEL_PATH)
        except Exception as e:
            print(f"WARNING: victim detection disabled ({e}).")
            victim_model = None

    # --- Setup ArUco Detector for Phase Switching ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    except AttributeError:
        aruco_params = cv2.aruco.DetectorParameters_create()
        detector = None
    
    cap = cv2.VideoCapture(video_path)
    
    # Global Position Variables
    global_x = 0.0 
    global_y = 0.0  
    current_z = Z0  
    
    # Velocity integrators
    current_vz = 0.0 
    vx_imu = 0.0
    vy_imu = 0.0
    
    trajectory_x = []
    trajectory_y = []
    altitude_log = []
    pose_log = [] # Array to store mapping telemetry
    center_colors = [] # Array to store dominant floor colors
    victim_hits = [] # projected victim detections, clustered after the pass
    
    # --- Live 2D Map Setup ---
    map_size = 800
    map_scale = 0.03 
    map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    map_center_x, map_center_y = map_size // 2, map_size // 2
    
    cv2.line(map_img, (map_center_x, 0), (map_center_x, map_size), (50, 50, 50), 1)
    cv2.line(map_img, (0, map_center_y), (map_size, map_center_y), (50, 50, 50), 1)
    
    cv2.putText(map_img, "N", (map_center_x - 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(map_img, "S", (map_center_x - 10, map_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(map_img, "E", (map_size - 25, map_center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(map_img, "W", (5, map_center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    prev_map_x = int(global_x * map_scale + map_center_x)
    prev_map_y = int(-global_y * map_scale + map_center_y) 
    
    old_gray = None
    old_points = None
    prev_yaw = None
    prev_az = None
    frame_idx = 0
    dt = 1.0 / FPS
    
    floor_mask = None
    wall_mask_cache = (-1, None)     # (frame_idx it belongs to, wall mask)
    saved_idx = 0                    # pose rows written so far (mapper stride)
    
    # State Machine Variables
    flight_phase = "TAKEOFF (MARKERS VISIBLE)"
    missing_marker_frames = 0
    
    print("\nStarting Visual Odometry Pipeline (Foveated Scale + Kinematic Bounding + Floor-Masked Features)...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # --- PRISTINE FRAME FOR MAPPING ---
        # Save a clean copy BEFORE any optical flow dots or UI text are drawn!
        clean_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- FLOOR MASK (recomputed every N frames) ---
        # One inference produces BOTH masks; the wall mask is cached for
        # the mapper so a frame is never inferred twice.
        if floor_model is not None and (floor_mask is None or
                                        frame_idx % MASK_EVERY_N == 0):
            floor_mask, _wm = infer_masks(floor_model, clean_frame)
            wall_mask_cache = (frame_idx, _wm)
        
        # --- PHASE DETECTION (ArUco Trigger) ---
        if detector:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            
        if ids is not None:
            missing_marker_frames = 0
            flight_phase = "TAKEOFF (MARKERS VISIBLE)"
            
            # --- ROBUST UPGRADE: ABSOLUTE ALTITUDE (ID-selected pair) ---
            z_marker = aruco_altitude(corners, ids)
            if z_marker is not None:
                current_z = z_marker
                
                if len(altitude_log) > 0:
                    current_vz = ((current_z - altitude_log[-1]) / 1000.0) / dt
        else:
            missing_marker_frames += 1
            if missing_marker_frames > 15:
                flight_phase = "CRUISE (RATE-OF-CHANGE LOCK)"
        
        # Time Sync
        current_time = frame_idx / FPS
        idx = (np.abs(imu_t - current_time)).argmin()
        
        current_yaw = imu_yaw[idx]
        current_ax = imu_ax[idx]
        current_ay = imu_ay[idx]
        current_az = imu_az[idx]
        current_wx = imu_wx[idx]
        current_wy = imu_wy[idx]
        
        if prev_yaw is None: prev_yaw = current_yaw
        if prev_az is None: prev_az = current_az
            
        # --- IMU KINEMATIC INTEGRATION ---
        rate_of_change_az = abs(current_az - prev_az)
        is_vertically_active = (rate_of_change_az > 0.05) or (abs(current_az) > 0.3)
        
        if flight_phase == "TAKEOFF (MARKERS VISIBLE)":
            z_imu_pred = current_z 
            vx_imu = 0.0
            vy_imu = 0.0
        else:
            if is_vertically_active:
                current_vz += current_az * dt
                current_vz *= 0.98 
            else:
                current_vz = 0.0
                
            delta_z_imu = current_vz * dt * 1000.0 
            z_imu_pred = current_z + delta_z_imu
            
            # Update Horizontal IMU Velocity (with relaxed decay to allow momentum)
            vx_imu = (vx_imu + current_ax * dt) * 0.995
            vy_imu = (vy_imu + current_ay * dt) * 0.995
        
        is_rotating = False
            
        # --- 1. Tracking Phase ---
        if old_gray is not None and old_points is not None and len(old_points) > 0:
            new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray, old_points, None, **lk_params)
            
            good_new = new_points[status == 1]
            good_old = old_points[status == 1]
            
            # --- EVICT FEATURES THAT WANDERED ONTO WALLS ---
            # (never below MIN_KEEP_POINTS: a few wall stragglers beat an
            # empty pool)
            if floor_mask is not None and len(good_new) > MIN_KEEP_POINTS:
                fx = np.clip(good_new[:, 0].astype(int), 0, floor_mask.shape[1] - 1)
                fy = np.clip(good_new[:, 1].astype(int), 0, floor_mask.shape[0] - 1)
                on_floor = floor_mask[fy, fx] > 0
                if on_floor.sum() >= MIN_KEEP_POINTS:
                    good_new = good_new[on_floor]
                    good_old = good_old[on_floor]
            
            if len(good_new) > 5:
                # --- RAW PIXEL MOVEMENT ---
                raw_dx_px = np.median(good_new[:, 0] - good_old[:, 0])
                raw_dy_px = np.median(good_new[:, 1] - good_old[:, 1])
                
                # --- ANALYTICAL GYRO COMPENSATION ---
                tilt_dx_px = (current_wx * dt * FOCAL_LENGTH) * GYRO_ROLL_SIGN
                tilt_dy_px = (current_wy * dt * FOCAL_LENGTH) * GYRO_PITCH_SIGN
                
                dx_px_total = raw_dx_px - tilt_dx_px
                dy_px_total = raw_dy_px - tilt_dy_px
                pixel_speed = np.sqrt(dx_px_total**2 + dy_px_total**2)
                
                # --- CENTER-FOVEATED ALTITUDE SCALING ---
                cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
                dist_to_center = np.linalg.norm(good_old - [cx, cy], axis=1)
                
                center_mask = dist_to_center < (min(cx, cy) * 0.4) 
                valid_old_z = good_old[center_mask]
                valid_new_z = good_new[center_mask]
                
                scale_ratio = 1.0
                valid_idx = np.array([])
                if len(valid_old_z) > 5:
                    centroid_old = np.mean(valid_old_z, axis=0)
                    centroid_new = np.mean(valid_new_z, axis=0)
                    dist_old_z = np.linalg.norm(valid_old_z - centroid_old, axis=1)
                    dist_new_z = np.linalg.norm(valid_new_z - centroid_new, axis=1)
                    
                    valid_idx = (dist_new_z > 1.0) & (dist_old_z > 1.0)
                    if np.sum(valid_idx) > 3:
                        scale_ratio = np.median(dist_old_z[valid_idx] / dist_new_z[valid_idx])
                        scale_ratio = np.clip(scale_ratio, 0.98, 1.02) 
                
                z_vis_meas = current_z * scale_ratio
                
                # --- Z-AXIS SENSOR FUSION ---
                if not Z_FROM_VISION:
                    # Consistent ruler: ArUco anchor + IMU vertical only.
                    if flight_phase != "TAKEOFF (MARKERS VISIBLE)":
                        current_z = z_imu_pred
                elif flight_phase == "TAKEOFF (MARKERS VISIBLE)":
                    pass 
                elif flight_phase == "CRUISE (RATE-OF-CHANGE LOCK)" and not is_vertically_active and pixel_speed > 1.5:
                    current_z = z_imu_pred
                elif pixel_speed > 1.5 or np.sum(valid_idx) <= 3:
                    current_z = (0.99 * z_imu_pred) + (0.01 * z_vis_meas)
                else:
                    current_z = (0.80 * z_imu_pred) + (0.20 * z_vis_meas)
                
                # --- TRANSLATION CALCULATIONS ---
                current_scale = FOCAL_LENGTH / current_z 
                
                delta_yaw = current_yaw - prev_yaw
                delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
                is_rotating = abs(delta_yaw) > ROTATION_THRESHOLD
                
                if not is_rotating:
                    # 1. Unbounded Visual Distance
                    dx_local_mm = -(dx_px_total / current_scale)
                    dy_local_mm = (dy_px_total / current_scale) 
                    
                    # 2. Expected IMU Distance
                    expected_dx_mm = vx_imu * dt * 1000.0
                    expected_dy_mm = vy_imu * dt * 1000.0
                    
                    # 3. KINEMATIC GATING (The Bounding Experiment)
                    # We check if the camera's measurement radically deviates from what physics dictates.
                    # Note: You may need to flip signs depending on Webots IMU vs Camera axis layout.
                    dx_error = dx_local_mm - expected_dx_mm
                    dy_error = dy_local_mm - expected_dy_mm
                    
                    if abs(dx_error) > MAX_DEVIATION_MM:
                        dx_local_mm = expected_dx_mm + (np.sign(dx_error) * MAX_DEVIATION_MM)
                        
                    if abs(dy_error) > MAX_DEVIATION_MM:
                        dy_local_mm = expected_dy_mm + (np.sign(dy_error) * MAX_DEVIATION_MM)

                    # 4. Global Rotation
                    dx_global_mm = dx_local_mm * np.cos(current_yaw) + dy_local_mm * np.sin(current_yaw)
                    dy_global_mm = -dx_local_mm * np.sin(current_yaw) + dy_local_mm * np.cos(current_yaw)
                    
                    global_x += dx_global_mm
                    global_y += dy_global_mm
                
                # Draw optical flow tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    color = (0, 165, 255) if is_rotating else (0, 255, 0)
                    cv2.line(frame, (int(a), int(b)), (int(c), int(d)), color, 2)
                    cv2.circle(frame, (int(a), int(b)), 3, color, -1)
            
            old_points = good_new.reshape(-1, 1, 2)
            
        # --- 2. Feature Replenishment Phase (FLOOR FIRST, FALLBACK UNMASKED) ---
        if old_points is None or len(old_points) < 50:
            new_features = cv2.goodFeaturesToTrack(gray, mask=floor_mask, **feature_params)
            if (new_features is None or len(new_features) < MIN_SPAWN_FLOOR) \
                    and floor_mask is not None:
                # Floor too small in this view (corridor) — spawn unmasked
                # rather than track nothing.
                new_features = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            if new_features is not None:
                if old_points is not None and len(old_points) > 0:
                    old_points = np.vstack((old_points, new_features))
                else:
                    old_points = new_features
                
                for pt in new_features:
                    cv2.circle(frame, (int(pt[0][0]), int(pt[0][1])), 4, (0, 0, 255), -1)
            
        old_gray = gray.copy()
        prev_yaw = current_yaw 
        prev_az = current_az
        
        trajectory_x.append(global_x)
        trajectory_y.append(global_y)
        altitude_log.append(current_z)

        # --- RECORD TELEMETRY & FRAMES FOR MAPPING ---
        # Save frames and pose logs strictly when not rotating to prevent motion blur in walls5.py
        if not is_rotating:
            frame_filename = f"frame_{frame_idx:06d}.jpg"
            frame_path = os.path.join(config.FRAMES_DIR, frame_filename)
            
            # Commit the pristine, spotless frame to disk! (No green dots)
            if SAVE_FRAMES:
                cv2.imwrite(frame_path, clean_frame)
            
            # Store critical pose elements needed to compute the projection matrix
            pose_row = {
                'frame_file': frame_filename,
                'global_x_mm': global_x,
                'global_y_mm': global_y,
                'altitude_z_mm': current_z,
                'yaw_rad': current_yaw
            }
            pose_log.append(pose_row)

            # --- FUSED VICTIM DETECTION: same frame, same pose ---
            # Runs on every non-rotating frame, matching the cadence
            # victims_location.py used (so MIN_DETECTION_HITS stays comparable).
            if victim_model is not None:
                v_h, v_w = clean_frame.shape[:2]
                drone_pos = (global_x, global_y, current_z)
                v_results = victim_model.predict(source=clean_frame,
                                                 classes=VICTIM_CLASSES,
                                                 conf=VICTIM_CONF, verbose=False)
                for v_res in v_results:
                    for box in v_res.boxes:
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                        if USE_BOX_FOOTPRINT:
                            gp = project_box_footprint(cx, cy, bx1, by1, bx2, by2,
                                                       drone_pos, current_yaw,
                                                       v_w, v_h)
                        else:
                            gp = image_to_global(cx, cy, drone_pos, current_yaw,
                                                 v_w, v_h)
                        if gp is not None:
                            gx_m, gy_m = gp[0] / 1000.0, gp[1] / 1000.0
                            # Parked ROSbots on the launch pad read as a "person"
                            # from directly overhead; never a real victim.
                            if np.hypot(gx_m, gy_m) < LAUNCH_EXCLUDE_M:
                                continue
                            vconf = float(box.conf[0].cpu().numpy())
                            C_m = (drone_pos[0] / 1000.0, drone_pos[1] / 1000.0,
                                   drone_pos[2] / 1000.0)
                            d_h = np.hypot(gx_m - C_m[0], gy_m - C_m[1])
                            H_m = max(0.1, C_m[2])
                            # cos^2 of the off-nadir angle: weight nadir views most.
                            nadir_w = (H_m * H_m) / (H_m * H_m + d_h * d_h)
                            victim_hits.append(dict(x_m=gx_m, y_m=gy_m, conf=vconf,
                                                    C=C_m, w=vconf * nadir_w))

            # --- FUSED MAPPING: feed this pose row straight to the mapper ---
            if mapper is not None:
                wm = None
                if saved_idx % FRAME_STRIDE == 0:
                    if wall_mask_cache[0] == frame_idx:
                        wm = wall_mask_cache[1]      # reuse this frame's inference
                    else:
                        _fm, wm = infer_masks(floor_model, clean_frame)
                mapper.feed(clean_frame if wm is not None else None,
                            pose_row, wm)
                if mapper.quit:
                    break
            saved_idx += 1
            
            # --- Sample Center Color for K-Means Clustering ---
            if frame_idx % 15 == 0:  # Sample every 15th frame
                h_f, w_f = clean_frame.shape[:2]
                cx, cy = w_f // 2, h_f // 2
                
                # Extract small patch from the clean frame, apply fast blur, and convert to LAB
                patch = clean_frame[cy-10:cy+10, cx-10:cx+10]
                blurred_patch = cv2.GaussianBlur(patch, (5, 5), 0)
                lab_patch = cv2.cvtColor(blurred_patch, cv2.COLOR_BGR2LAB)
                
                avg_color = np.mean(lab_patch, axis=(0, 1))
                center_colors.append(avg_color)

        # --- Update Live Maps & UI ---
        curr_map_x = int(global_x * map_scale + map_center_x)
        curr_map_y = int(-global_y * map_scale + map_center_y) 
        
        cv2.line(map_img, (prev_map_x, prev_map_y), (curr_map_x, curr_map_y), (255, 0, 0), 2)
        prev_map_x, prev_map_y = curr_map_x, curr_map_y
        
        display_map = map_img.copy()
        
        arrow_len = 30
        heading_x = int(curr_map_x + arrow_len * np.sin(current_yaw))
        heading_y = int(curr_map_y - arrow_len * np.cos(current_yaw))
        cv2.arrowedLine(display_map, (curr_map_x, curr_map_y), (heading_x, heading_y), (0, 0, 255), 2, tipLength=0.3)
        
        cv2.circle(display_map, (curr_map_x, curr_map_y), 6, (0, 255, 0), -1)
        cv2.putText(display_map, f"Pos: {global_x/1000:.2f}m, {global_y/1000:.2f}m", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Phase: {flight_phase}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 255), 2)
        cv2.putText(frame, f"Fused Alt: {current_z/1000:.3f} m", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Pos X: {global_x/1000:.2f} m", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Pos Y: {global_y/1000:.2f} m", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if is_rotating:
            cv2.putText(frame, "ROTATION DETECTED: Pausing Translation", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        if SHOW_ODOM:
            cv2.imshow("Drone Visual Odometry", frame)
            cv2.imshow("Live 2D Trajectory", display_map)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # --- FUSED MAPPING: flush submaps, sweep footprint, save wall_raw.npy ---
    if mapper is not None:
        mapper.finish()

    # --- SAVE MAPPING TELEMETRY ---
    print("\nExporting dedicated mapping telemetry registry to 'mapping_data/camera_poses.csv'...")
    poses_df = pd.DataFrame(pose_log)
    poses_df.to_csv(config.CSV_PATH, index=False)

    # --- FUSED VICTIM DETECTION: cluster the pass's hits, write the CSV ---
    if DETECT_VICTIMS:
        finalize_victims(victim_hits)

    # --- COMPUTE AND SAVE DOMINANT COLOR PALETTES ---
    if len(center_colors) > 0:
        print("\nComputing dominant floor/table color palettes (K=5)...")
        center_colors_array = np.float32(center_colors)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(center_colors_array, 7, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Save to numpy format so the edge detector can load it instantly
        np.save(os.path.join(config.MAPPING_DIR, 'dominant_palettes.npy'), centers)
        print("Saved dominant palettes to %s"
              % os.path.join(config.MAPPING_DIR, 'dominant_palettes.npy'))
        for i, c in enumerate(centers):
            print(f"Palette {i+1} (LAB): {c}")

    if trajectory_x and trajectory_y:
        print("\nGenerating verification maps...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        traj_x_m = [x / 1000.0 for x in trajectory_x]
        traj_y_m = [y / 1000.0 for y in trajectory_y]
        
        ax1.plot(traj_x_m, traj_y_m, label='Drone Trajectory', color='blue', linewidth=2)
        ax1.scatter(traj_x_m[0], traj_y_m[0], color='green', s=100, label='Start Position', zorder=5)
        ax1.scatter(traj_x_m[-1], traj_y_m[-1], color='red', s=100, label='End Position', zorder=5)
        ax1.set_title('2D Drone Flight Path')
        ax1.set_xlabel('Global East (meters)')
        ax1.set_ylabel('Global North (meters)')
        ax1.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.85), arrowprops=dict(facecolor='black', width=3, headwidth=10), ha='center', va='center', fontsize=14, fontweight='bold', xycoords='axes fraction', textcoords='axes fraction')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.axhline(0, color='black', linewidth=1)
        ax1.axvline(0, color='black', linewidth=1)
        ax1.legend(loc='lower right')
        ax1.axis('equal') 
        
        alt_m = [z / 1000.0 for z in altitude_log]
        time_s = [i / FPS for i in range(len(alt_m))]
        
        ax2.plot(time_s, alt_m, label='IMU/Visual Fused Altitude', color='purple', linewidth=2)
        ax2.set_title('Z-Axis Altitude Verification')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Altitude (meters)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        fig.savefig(os.path.join(config.MAPPING_DIR, "trajectory_verification.png"),
                    dpi=120)
        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)






"""STAGE 3 -- the ROSE spectral filter. The pipeline stops here.

Implements ROSE (Kucner, Luperto et al., arXiv:2004.08794 and
arXiv:2203.03519). The method is theirs; this is our own implementation of
it, written from the papers and adapted for aerial flyover footage. Cite
their work, not this file. Reference implementation:
https://github.com/aislabunimi/ROSE2

Walls in a built environment are long, straight and nearly all parallel to one
of a few directions, so in the 2D Fourier transform of the occupancy map they
concentrate into a few angular ridges. Clutter, reflections and pose-drift smear
have no such preference and spread evenly across all angles. That difference is
the whole filter: keep only wedges around the strongest ridges, transform back,
then apply a structure cut and an intensity cut.

The result is the denoised occupancy grid. There is no vectorisation here by
design -- no Hough fragments, no wall fitting, no door carving, no single-line
floor plan. The grid goes in and the same grid comes out with the
non-structural occupancy removed.
"""

import math
import os

import cv2
import numpy as np

from . import config
from .config import *  # noqa: F401,F403

def wrap_pi(a):
    """wrap an angle into [0, pi)"""
    return a % math.pi



def rose_intensity(raw):
    """Occupancy -> spectral wedge filter -> structure cut -> intensity cut.
    Returns the binary wall map (r3b) and the dominant wall directions."""
    b = (raw > 0).astype(np.float64)
    F = np.fft.fftshift(np.fft.fft2(b))
    amp = np.abs(F)

    h, w = b.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    ang = (np.degrees(np.arctan2(yy - cy, xx - cx))) % 180.0
    rad = np.hypot(yy - cy, xx - cx)

    sel = rad > LOW_FREQ_KEEP
    hist = np.zeros(180)
    np.add.at(hist, ang[sel].astype(int) % 180, amp[sel])
    hist = cv2.GaussianBlur(hist.reshape(-1, 1), (1, 5), 0).ravel()
    peaks = []
    for a in np.argsort(hist)[::-1]:
        if all(min(abs(a - p), 180 - abs(a - p)) > 30 for p in peaks):
            peaks.append(int(a))
        if len(peaks) >= N_DIRECTIONS:
            break

    mask = rad <= LOW_FREQ_KEEP
    for p in peaks:
        d = np.minimum(np.abs(ang - p), 180 - np.abs(ang - p))
        mask |= d <= WEDGE_DEG
    resp = np.maximum(np.real(np.fft.ifft2(np.fft.ifftshift(F * mask))), 0)

    thresh = np.quantile(resp[raw > 0], 1.0 - KEEP_FRACTION)
    structure = ((raw > 0) & (resp >= thresh)).astype(np.uint8) * 255

    win = max(1, INT_WIN | 1)
    intensity = cv2.boxFilter(resp * (structure > 0), -1, (win, win),
                              normalize=False)
    s = structure > 0
    ithresh = float(np.quantile(intensity[s], INT_Q)) if s.any() else 0.0
    binary = (s & (intensity >= ithresh)).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(config.TUNING_DIR, "r3b_intensity_binary.png"), binary)
    print("[rose] ridges %s deg | raw %d -> structure %d -> binary %d px"
          % (peaks, int((raw > 0).sum()), int(s.sum()), int((binary > 0).sum())))

    wall_dirs = [wrap_pi(math.radians(p) + math.pi / 2) for p in peaks]
    return binary, wall_dirs


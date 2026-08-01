# 2d_recon_from_3d

2D wall reconstruction from a 3D flyover. Give it a video and its IMU log, get
back the building's wall structure as a denoised occupancy map.

```
video + IMU csv
    -> STAGE 1  visual odometry          camera trajectory
    -> STAGE 2  wall grid from frames    raw occupancy accumulation
    -> STAGE 3  ROSE spectral filter     the denoised wall map     <- output
```

The output is the **frequency-filtered occupancy map**, not a vectorised floor
plan. The pipeline stops at `rose_intensity()`: no Hough fragments, no wall
fitting, no door carving, no single-line rendering. What comes out is the same
grid that went in, with everything that does not belong to the building's
dominant wall directions removed.

---

## Install

Python 3.10+.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Install **only** `opencv-contrib-python`. Having it and `opencv-python` in the
same environment can shadow `cv2.aruco`, which stage 1 needs to fix scale and
the reference frame from the origin-marker pattern.

`models/best_segmentation.pt` is the wall/floor segmentation model and ships
with the project.

---

## Layout

```
recon.py                  run without installing
rose_recon/
    __init__.py           public API: configure, reconstruct, rose_intensity
    __main__.py           python -m rose_recon
    cli.py                argument parsing
    config.py             tuning constants + per-run paths
    perception.py         model loading, floor and wall masks   (shared)
    odometry.py           STAGE 1  visual odometry
    mapping.py            STAGE 2  wall grid accumulation
    rose.py               STAGE 3  the spectral filter
    pipeline.py           runs the stages, writes before/after
models/best_segmentation.pt
```

Dependencies run one way only: `config` <- `perception` <- `odometry`,
`mapping` <- `pipeline`. `perception.py` exists because both stages need the
segmentation model, and putting those helpers in either stage would make the two
import each other.

Paths are the one piece of shared mutable state, so they are handled carefully:
`config.configure()` fills them in per run, and every module reaches them as
`config.WALL_NPY` rather than importing the bare name. They are excluded from
`config.__all__` for that reason — a reference that forgets the prefix raises
NameError immediately instead of silently writing to the wrong place.

---

## Run

```bash
python recon.py --video path/to/flyover.mp4
# or, equivalently
python -m rose_recon --video path/to/flyover.mp4
```

As a library:

```python
import rose_recon

rose_recon.configure(video="flyover.mp4")
binary, wall_dirs = rose_recon.reconstruct()
```

Or filter an occupancy grid you already have, skipping stages 1 and 2 entirely:

```python
from rose_recon import rose_intensity

binary, wall_dirs = rose_intensity(raw_occupancy)   # uint8, 0 or 255
```

The IMU csv defaults to the video path with a `.csv` extension, which is how
recordings are normally shipped. Override it with `--imu` if not.

| Option | Effect |
|---|---|
| `--video <mp4>` | flyover footage to reconstruct (required) |
| `--imu <csv>` | inertial log; defaults to the video path with `.csv` |
| `--out <dir>` | output folder; defaults to `output/<video name>/` |
| `--force` | redo the video pass even if a trajectory is cached |
| `--force-map` | redo the wall accumulation even if a grid is cached |
| `--quiet` | no live odometry windows |

Results are cached per video. The first run does the full video pass and takes
a few minutes; re-running only redoes stage 3, which takes about a second. Two
videos never overwrite each other's cache.

### Output, under `output/<video name>/`

| File | What it is |
|---|---|
| `walls_rose.png` | **the deliverable** — wall map after the spectral filter |
| `walls_before.png` | raw accumulation, straight out of stage 2 |
| `wall_raw.npy` | that raw accumulation as an array |
| `camera_poses.csv` | recovered trajectory |
| `tuning/` | stage-3 intermediates |

Both PNGs are white-on-black at the grid's own resolution, so they line up
pixel for pixel and can be diffed directly.

---

## How stage 3 works

Walls in a built environment are long, straight, and nearly all parallel to one
of a small number of directions. In the 2D Fourier transform of the occupancy
map they therefore concentrate into a few angular ridges. Clutter, reflections
and pose-drift smear have no such preference and spread evenly across all
angles. That difference is the whole filter.

1. **Transform.** FFT the binarised occupancy map.
2. **Find the wall directions.** Build an angular histogram of amplitude over
   everything above `LOW_FREQ_KEEP`, blur it, and take the `N_DIRECTIONS`
   strongest peaks at least 30 degrees apart. For a rectilinear building these
   come out near 0 and 90 degrees.
3. **Keep only those wedges.** Mask the spectrum to +/- `WEDGE_DEG` around each
   peak, plus the low frequencies that carry the building's overall shape, and
   transform back. Structure is rebuilt; noise is not.
4. **Structure cut.** Of the cells that were occupied to begin with, keep the
   strongest `KEEP_FRACTION` by spectral response.
5. **Intensity cut.** Box-filter the response over the survivors and drop those
   below the `INT_Q` quantile, which removes isolated specks that survived the
   wedge but sit in no dense neighbourhood.

On a typical flyover this removes about half the occupied cells while leaving
the walls intact.

### Tuning

All in `rose_recon/config.py`.

| Setting | Effect |
|---|---|
| `N_DIRECTIONS` | how many wall directions the building has. 2 is rectilinear; raise it for non-orthogonal layouts |
| `WEDGE_DEG` | half-width kept around each direction. Wider tolerates skew and drift, narrower cuts more clutter |
| `LOW_FREQ_KEEP` | radius of low frequencies passed through untouched |
| `KEEP_FRACTION` | structure cut. Lower keeps less |
| `INT_Q` | intensity cut. Higher removes more isolated survivors |

If the result looks over-cut, raise `KEEP_FRACTION` and lower `INT_Q` first —
those two do most of the work. If walls are missing entirely, check the ridge
angles printed by stage 3 before touching anything else: they should match the
building, and if they do not, the trajectory is drifting and the problem is
upstream in stage 1.

---

## Notes

Stage 1 can also detect people on the same video pass. That is irrelevant here,
so `DETECT_VICTIMS` in `config.py` is off and the person-detection weights are never loaded or
required.

`odometry.py`, `mapping.py` and `rose_intensity()` are lifted unchanged from the SAR
flyover pipeline they were developed in. Only the configuration and the entry
point are new, so the reconstruction behaves identically to that pipeline up to
the point where it stops.

---

## License

MIT. See [LICENSE](LICENSE).

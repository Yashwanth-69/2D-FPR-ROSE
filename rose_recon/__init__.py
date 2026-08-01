"""2D wall reconstruction from a 3D flyover.

    video + IMU csv
        -> STAGE 1  visual odometry          camera trajectory
        -> STAGE 2  wall grid from frames    raw occupancy accumulation
        -> STAGE 3  ROSE spectral filter     the denoised wall map

The output is the frequency-filtered occupancy map. The pipeline stops at
rose_intensity(): no Hough fragments, no wall fitting, no door carving, no
single-line rendering.

    import rose_recon
    rose_recon.configure(video="flyover.mp4")
    binary, wall_dirs = rose_recon.reconstruct()

Or filter a grid you already have:

    from rose_recon.rose import rose_intensity
    binary, wall_dirs = rose_intensity(raw_occupancy)
"""

from .config import configure
from .pipeline import reconstruct, save_before_after
from .rose import rose_intensity

__all__ = ["configure", "reconstruct", "save_before_after", "rose_intensity"]
__version__ = "1.0.0"

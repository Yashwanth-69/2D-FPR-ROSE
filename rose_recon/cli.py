"""Command line front end."""

import argparse
import sys

from . import config
from .pipeline import reconstruct


def build_parser():
    p = argparse.ArgumentParser(
        prog="rose_recon",
        description="2D wall reconstruction from a 3D flyover. Outputs the "
                    "frequency-filtered occupancy map, not a vectorised floor plan.")
    p.add_argument("--video", "-v", required=True,
                   help="flyover footage to reconstruct")
    p.add_argument("--imu", "-i", default=None,
                   help="inertial log (default: the video path with a .csv extension)")
    p.add_argument("--out", "-o", default=None,
                   help="output folder (default: output/<video name>/)")
    p.add_argument("--model", default=None,
                   help="wall/floor segmentation weights (default: the bundled model)")
    p.add_argument("--force", action="store_true",
                   help="redo the video pass even if a trajectory is cached")
    p.add_argument("--force-map", action="store_true", dest="force_map",
                   help="redo the wall accumulation even if a grid is cached")
    p.add_argument("--quiet", action="store_true",
                   help="no live odometry windows")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    config.configure(video=a.video, imu=a.imu, out=a.out, model=a.model,
                     force=a.force, force_map=a.force_map, show=not a.quiet)
    reconstruct()
    return 0


if __name__ == "__main__":
    sys.exit(main())

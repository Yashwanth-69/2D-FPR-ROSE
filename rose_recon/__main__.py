"""Enables `python -m rose_recon --video flyover.mp4`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Thin entry point: `python recon.py --video flyover.mp4`.

Identical to `python -m rose_recon`. The real code lives in the rose_recon
package; this exists so the project can be run without installing it.
"""

import sys

from rose_recon.cli import main

if __name__ == "__main__":
    sys.exit(main())

"""Small targeted grid for hard second grasps; never imported by policy."""

import sys

from probe_relative_second_grasp import run


if __name__ == "__main__":
    seed = int(sys.argv[1])
    x = float(sys.argv[2])
    y = float(sys.argv[3])
    print(seed, x, y, run(seed, x, y), flush=True)

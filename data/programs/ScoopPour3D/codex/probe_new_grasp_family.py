"""Reproduce the long-alignment canonical grasp found for seeds 6 and 8.

This deliberately remains an exploration script: approach.py never imports it.
The important difference from the original canonical probe is allowing the
slow simulated base enough steps to converge before descending.
"""

import argparse

from probe_scoop_grasp_search import trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--wrist", type=float, default=0.0,
                        help="additional wrist delta; seed 9 grasps at -0.4")
    args = parser.parse_args()
    args.canonical = True
    args.long = 0.06
    args.short = 0.0
    args.depth = 32
    args.elbow = 0.06
    args.close_early = 2
    args.open_cmd = 1.0
    args.close_cmd = 0.0
    args.align_steps = 40
    args.close_hold = 8
    args.lift_steps = 18
    args.verify_y = 0.06
    args.verify_steps = 4
    trial(args)


if __name__ == "__main__":
    main()

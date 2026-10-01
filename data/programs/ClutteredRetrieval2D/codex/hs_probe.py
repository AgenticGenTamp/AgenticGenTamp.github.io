"""Independent black-box probes for ClutteredRetrieval2D (heuristic_solver agent)."""

import argparse
import math
import numpy as np

from env_client import make_env


def val(s, o, f):
    return float(s.get(o, f))


def summarize(s):
    out = []
    for name in s.get_object_names():
        o = s.get_object_from_name(name)
        # Infer available features by trying the common ones.
        fields = {}
        for f in ("x", "y", "theta", "width", "height", "base_radius",
                  "arm_joint", "arm_length", "vacuum", "gripper_height",
                  "gripper_width", "static"):
            try:
                fields[f] = round(val(s, o, f), 4)
            except Exception:
                pass
        out.append((name, fields))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--count", type=int)
    p.add_argument("--action", nargs=5, type=float)
    p.add_argument("--steps", type=int, default=1)
    args = p.parse_args()
    env = make_env()
    options = None if args.count is None else {"object_count": args.count}
    s, info = env.reset(seed=args.seed, options=options)
    print("reset", args.seed, args.count, "info", info, "max", env.max_steps)
    print(*summarize(s), sep="\n")
    if args.action:
        a = np.asarray(args.action, dtype=np.float32)
        for i in range(args.steps):
            s, r, term, trunc, info = env.step(a)
            print("step", i + 1, r, term, trunc, info)
            print(*summarize(s), sep="\n")
            if term or trunc:
                break
    env.close()


if __name__ == "__main__":
    main()

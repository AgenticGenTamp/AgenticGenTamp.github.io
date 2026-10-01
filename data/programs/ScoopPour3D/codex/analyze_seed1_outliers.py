"""Diagnose seed-1 translated-target outliers at the middle of the sweep.

This deliberately reports every cube (rather than assuming a fixed count) and
relates final error to its initial source-bin coordinates.  It is a diagnostic
only: approach.py is not imported or modified indirectly beyond normal use.
"""

import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def xyz(state, name):
    obj = state.get_object_from_name(name)
    return np.asarray([state.get(obj, f) for f in ("x", "y", "z")], float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--step", type=int, default=270)
    args = parser.parse_args()

    env = make_env()
    state, info = env.reset(seed=args.seed, options={"object_count": args.count})
    names = sorted(n for n in state.get_object_names() if n.startswith("cube_"))
    initial = np.asarray([xyz(state, n) for n in names])
    source0 = xyz(state, "bin_yellow_0")
    target0 = xyz(state, "bin_green_0")
    translated = initial + target0 - source0

    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    reward = 0.0
    for _ in range(args.step):
        state, reward, terminated, truncated, _ = env.step(
            policy.get_action(state)
        )
        if terminated or truncated:
            break

    final = np.asarray([xyz(state, n) for n in names])
    delta = final - translated
    err3 = np.linalg.norm(delta, axis=1)
    errxy = np.linalg.norm(delta[:, :2], axis=1)
    init_rel = initial - source0
    order = np.argsort(-errxy)

    print("step", args.step, "reward", reward,
          "source", np.round(source0, 4), "target", np.round(target0, 4))
    print("name init_rel_x init_rel_y final_x final_y dx dy dz err_xy err_3d")
    for i in order:
        print("%-8s %+.4f %+.4f %.4f %.4f %+.4f %+.4f %+.4f %.4f %.4f" %
              (names[i], init_rel[i, 0], init_rel[i, 1], final[i, 0],
               final[i, 1], delta[i, 0], delta[i, 1], delta[i, 2],
               errxy[i], err3[i]))

    for axis, label in ((0, "initial x"), (1, "initial y")):
        if len(names) > 1:
            print("corr(error_xy, %s)" % label,
                  round(float(np.corrcoef(errxy, init_rel[:, axis])[0, 1]), 4),
                  "corr(dx, %s)" % label,
                  round(float(np.corrcoef(delta[:, 0], init_rel[:, axis])[0, 1]), 4),
                  "corr(dy, %s)" % label,
                  round(float(np.corrcoef(delta[:, 1], init_rel[:, axis])[0, 1]), 4))
    q = np.quantile(errxy, 0.75)
    out = errxy >= q
    print("upper-quartile outliers", [names[i] for i in np.flatnonzero(out)])
    print("outlier mean init_rel", np.round(init_rel[out, :2].mean(0), 4),
          "inlier", np.round(init_rel[~out, :2].mean(0), 4))
    print("mean signed correction needed (x,y)",
          np.round(-delta[:, :2].mean(0), 4),
          "outliers", np.round(-delta[out, :2].mean(0), 4))

    # Fit the smallest rigid planar correction to the whole pile.  For a
    # small rotation theta about the final pile centroid:
    # correction_x = tx - theta * rel_y and
    # correction_y = ty + theta * rel_x.
    correction = translated[:, :2] - final[:, :2]
    rel = final[:, :2] - final[:, :2].mean(0)
    design = np.zeros((2 * len(names), 3))
    rhs = correction.reshape(-1)
    design[0::2, 0] = 1.0
    design[0::2, 2] = -rel[:, 1]
    design[1::2, 1] = 1.0
    design[1::2, 2] = rel[:, 0]
    tx, ty, theta = np.linalg.lstsq(design, rhs, rcond=None)[0]
    before = float(np.sqrt(np.mean(np.sum(correction ** 2, axis=1))))
    residual = rhs - design @ np.array([tx, ty, theta])
    after = float(np.sqrt(np.mean(residual.reshape(-1, 2) ** 2) * 2.0))
    print("best rigid correction tx ty yaw(rad)",
          np.round([tx, ty, theta], 5), "rms", round(before, 5), "->",
          round(after, 5))
    print("preserving command: a[0]=clip(0.2*tx,+/-0.002),",
          "retain existing a[1] y feedback; a[2]=clip(0.2*yaw,+/-0.001).",
          "Omit yaw when |yaw|<0.01 or rigid-fit RMS does not improve >=20%.")
    env.close()


if __name__ == "__main__":
    main()

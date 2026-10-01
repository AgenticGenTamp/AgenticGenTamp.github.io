"""Random low-arm pose search; reports the first physical cube contact."""

import numpy as np
from env_client import make_env


def vals(s, name, fs):
    o = s.get_object_from_name(name)
    return np.array([float(s.get(o, f)) for f in fs])


def run(seed=0):
    rng = np.random.default_rng(88421 + seed)
    env = make_env()
    s, _ = env.reset(seed=seed, options={"object_count": 1})
    c0 = vals(s, "cube_0", ("x", "y", "z"))
    # Start from a normal, collision-free pose but put the cube at likely arm reach.
    offsets = [(0.50, 0.0), (0.35, 0.0), (0.65, 0.0),
               (0.50, 0.18), (0.50, -0.18)]
    home = np.array([0., -.349, np.pi, -2.548, 0., -.873, np.pi / 2])
    found = None
    for trial in range(24):
        # Bias toward elbow-down poses while randomizing all wrist/azimuth joints.
        q = np.array([rng.uniform(-1.2, 1.2), rng.uniform(-0.2, 1.35),
                      rng.uniform(1.8, 4.45), rng.uniform(-2.8, 0.1),
                      rng.uniform(-1.5, 1.5), rng.uniform(-2.5, 0.2),
                      rng.uniform(0., 3.15)])
        off = offsets[trial % len(offsets)]
        base_goal = np.array([c0[0] - off[0], c0[1] - off[1], 0.0])
        for k in range(34):
            b = vals(s, "robot", ("pos_base_x", "pos_base_y", "pos_base_rot"))
            cq = vals(s, "robot", tuple("pos_arm_joint%d" % i for i in range(1, 8)))
            a = np.zeros(18, np.float32)
            a[:3] = np.clip(base_goal - b, -.1, .1)
            a[3:10] = np.clip(q - cq, -.1, .1)
            a[10] = 0.
            a[11:18] = np.clip(3 * (q - cq), -2., 2.)
            s, _, term, trunc, _ = env.step(a)
            c = vals(s, "cube_0", ("x", "y", "z"))
            shift = np.linalg.norm(c - c0)
            if shift > .002:
                found = (trial, k, off, q, cq, c0, c, shift)
                break
            if term or trunc:
                break
        if found:
            break
        # Return high between random probes to make subsequent paths diverse.
        for _ in range(8):
            cq = vals(s, "robot", tuple("pos_arm_joint%d" % i for i in range(1, 8)))
            a = np.zeros(18, np.float32)
            a[3:10] = np.clip(home - cq, -.1, .1)
            s, _, _, trunc, _ = env.step(a)
            if trunc:
                break
    if found:
        trial, k, off, q, beforeq, old, new, shift = found
        print("FOUND", seed, trial, k, "off", off, "shift", shift,
              "qtarget", np.round(q, 4), "qbefore", np.round(beforeq, 4),
              "cube", np.round(old, 4), np.round(new, 4), flush=True)
    else:
        print("NONE", seed, "final", np.round(vals(s, "cube_0", ("x", "y", "z")), 4), flush=True)
    env.close()


if __name__ == "__main__":
    for sd in range(3):
        run(sd)

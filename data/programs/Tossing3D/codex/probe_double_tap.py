"""Measure whether a rapid arm reset and second calibrated strike adds lift."""
import sys
import numpy as np
from env_client import make_env

FS = tuple("pos_arm_joint%d" % i for i in range(1, 8))
Q = np.array([.9441, -.0616, 1.4739, 1.3419, -.5038, -1.5938, .2053])
ALPHA = -1.305
CA, SA = np.cos(ALPHA), np.sin(ALPHA)
R = np.array([[CA, -SA], [SA, CA]])
OFF = R @ np.array([.6, .25])
ACTS = np.load("random_contact_found.npz")["actions"][139:145]


def v(s, name, fs):
    o = s.get_object_from_name(name)
    return np.array([float(s.get(o, f)) for f in fs])


def run(reset_steps, strength):
    env = make_env()
    s, _ = env.reset(seed=0, options={"object_count": 1})
    peak = v(s, "cube_0", ("z",))[0]
    # Exactly reproduce the initial prepose used by the working policy.
    for _ in range(55):
        p = v(s, "cube_0", ("x", "y")); b = v(s, "robot", ("pos_base_x", "pos_base_y"))
        q = v(s, "robot", FS); a = np.zeros(18, np.float32)
        a[:2] = np.clip(p - OFF - b, -.06, .06)
        a[2] = np.clip(.4799 + ALPHA - v(s, "robot", ("pos_base_rot",))[0], -.06, .06)
        a[3:10] = np.clip(Q - q, -.06, .06); a[10] = 1.
        a[11:] = np.clip(4 * (Q - q), -4, 4)
        s, _, _, _, _ = env.step(a)
        ye = (.4799 + ALPHA - v(s, "robot", ("pos_base_rot",))[0] + np.pi) % (2*np.pi) - np.pi
        if (np.linalg.norm(p - OFF - v(s, "robot", ("pos_base_x", "pos_base_y"))) < .018
                and abs(ye) < .025 and np.max(np.abs(Q-v(s, "robot", FS))) < .035):
            break
    samples = []
    for tag, seq in (("first", ACTS),):
        for a0 in seq:
            a = a0.copy(); a[:2] = R @ a[:2]; a[10] = 1.
            s, _, _, _, _ = env.step(a)
            z = v(s, "cube_0", ("z",))[0]; peak = max(peak, z); samples.append((tag, z))
    # Command directly back toward the calibrated prepose, without waiting for convergence.
    for _ in range(reset_steps):
        p = v(s, "cube_0", ("x", "y")); b = v(s, "robot", ("pos_base_x", "pos_base_y"))
        q = v(s, "robot", FS); a = np.zeros(18, np.float32); a[10] = 1.
        a[:2] = np.clip(p - OFF - b, -.06, .06)
        a[2] = np.clip(.4799 + ALPHA - v(s, "robot", ("pos_base_rot",))[0], -.06, .06)
        a[3:10] = np.clip(Q - q, -.1, .1)
        a[11:] = np.clip(strength * (Q - q), -12, 12)
        s, _, _, _, _ = env.step(a)
        z = v(s, "cube_0", ("z",))[0]; peak = max(peak, z); samples.append(("reset", z))
    for a0 in ACTS:
        a = a0.copy(); a[:2] = R @ a[:2]; a[10] = 1.
        s, _, _, _, _ = env.step(a)
        z = v(s, "cube_0", ("z",))[0]; peak = max(peak, z); samples.append(("second", z))
    for _ in range(12):
        a = np.zeros(18, np.float32); a[10] = 1.
        s, _, _, _, _ = env.step(a)
        z = v(s, "cube_0", ("z",))[0]; peak = max(peak, z); samples.append(("coast", z))
    print(reset_steps, strength, "peak", round(peak, 5), "final", np.round(v(s, "cube_0", ("x", "y", "z")), 4),
          "zs", [round(x[1], 4) for x in samples], flush=True)
    env.close()


if __name__ == "__main__":
    run(int(sys.argv[1]), float(sys.argv[2]))

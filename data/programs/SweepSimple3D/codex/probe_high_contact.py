"""Random broad/folded arm contact search with a stationary, distant base."""
import math
import numpy as np
from env_client import make_env


def val(s, name, feat):
    return float(s.get(s.get_object_from_name(name), feat))


def pos(s, name):
    return np.array([val(s, name, f) for f in "xyz"])


def base(s):
    return np.array([val(s, "robot", "pos_base_x"), val(s, "robot", "pos_base_y")])


def joints(s):
    return np.array([val(s, "robot", f"pos_arm_joint{i}") for i in range(1, 8)])


def angle_error(target, actual):
    return (target - actual + math.pi) % (2 * math.pi) - math.pi


def place_base(e, s, bp, yaw):
    # Leave the arm at home and open so base motion cannot masquerade as arm contact.
    for _ in range(28):
        a = np.zeros(11, np.float32)
        a[:2] = np.clip(.75 * (bp - base(s)), -.1, .1)
        a[2] = np.clip(.75 * angle_error(yaw, val(s, "robot", "pos_base_rot")), -.1, .1)
        a[10] = 1.
        s, _, _, _, _ = e.step(a)
    for _ in range(5):
        a = np.zeros(11, np.float32); a[10] = 1.
        s, _, _, _, _ = e.step(a)
    return s


def main():
    rng = np.random.default_rng(1853)
    e = make_env()
    # Bearings exercise front, oblique, and side reach, always > 35 cm.
    cases = [(.43, math.pi/2), (.50, 3*math.pi/4), (.43, math.pi),
             (.55, math.pi/4)]
    # Joint 4 stays elbow-folded; shoulder and wrist spans are deliberately broad.
    lo = np.array([-2.8, -1.35, .35, -2.95, -2.8, -1.75, -2.8])
    hi = np.array([ 2.8,  1.35, 5.85,  -.35,  2.8,  1.10,  2.8])
    for case, (radius, bearing) in enumerate(cases):
        s, _ = e.reset(seed=0, options={"object_count": 1})
        w0 = pos(s, "wiper_0")
        bp = w0[:2] + radius*np.array([math.cos(bearing), math.sin(bearing)])
        yaw = angle_error(bearing + math.pi, 0.)
        s = place_base(e, s, bp, yaw)
        settled = pos(s, "wiper_0")
        print("CASE", case, "base", np.round(base(s), 4).tolist(),
              "radius", round(float(np.linalg.norm(base(s)-settled[:2])), 4),
              "yaw", round(val(s,"robot","pos_base_rot"),4),
              "wsettle", np.round(settled-w0, 4).tolist(), flush=True)
        previous = settled.copy()
        for trial in range(10):
            target = rng.uniform(lo, hi)
            # Alternating high shoulder postures improves vertical workspace coverage.
            if trial % 2 == 0:
                target[1] = rng.uniform(.35, 1.35)
            for step in range(30):
                a = np.zeros(11, np.float32)
                a[3:10] = np.clip(.72*(target-joints(s)), -.1, .1)
                a[10] = 1.
                qb = joints(s).copy(); wb = pos(s, "wiper_0")
                s, _, _, _, _ = e.step(a)
                wa = pos(s, "wiper_0")
                d = wa-wb
                if np.linalg.norm(d) > .002:
                    print("CONTACT case/trial/step", case, trial, step,
                          "base", np.round(base(s),4).tolist(),
                          "yaw", round(val(s,"robot","pos_base_rot"),4),
                          "q_before", np.round(qb,4).tolist(),
                          "q_after", np.round(joints(s),4).tolist(),
                          "target", np.round(target,4).tolist(),
                          "w_before", np.round(wb,4).tolist(),
                          "w_delta", np.round(d,4).tolist(), flush=True)
                previous = wa
        print("END", case, "w_total", np.round(pos(s,"wiper_0")-settled,4).tolist(), flush=True)
    e.close()


if __name__ == "__main__":
    main()

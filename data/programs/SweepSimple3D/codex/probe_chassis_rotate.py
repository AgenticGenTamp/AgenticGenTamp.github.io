"""Probe a two-stage chassis-only blade rotation and southward sweep."""
import sys
import math
import numpy as np
from env_client import make_env


def val(s, name, feature):
    return float(s.get(s.get_object_from_name(name), feature))


def xy(s, name):
    return np.array([val(s, name, "x"), val(s, name, "y")])


def base(s):
    return np.array([val(s, "robot", "pos_base_x"), val(s, "robot", "pos_base_y")])


def yaw(s):
    qw, qx, qy, qz = [val(s, "wiper_0", f) for f in ("qw", "qx", "qy", "qz")]
    return math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))


def run(offset, seed=0, count=5):
    env = make_env()
    s, _ = env.reset(seed=seed, options={"object_count": count})
    cubes = sorted(n for n in s.get_object_names() if n.startswith("cube_"))
    init = {n: xy(s, n) for n in ["wiper_0"] + cubes}
    steps = 0

    def drive(goal, limit, gain=1.2, cap=.1):
        nonlocal s, steps
        for _ in range(limit):
            d = goal - base(s)
            if np.linalg.norm(d) < .025:
                break
            a = np.zeros(11, np.float32)
            a[:2] = np.clip(gain*d, -cap, cap)
            s, r, term, trunc, _ = env.step(a)
            steps += 1
            if r > -1 or term:
                print("EVENT", offset, steps, r, term, base(s), xy(s,"wiper_0"), flush=True)
            if term or trunc:
                break

    w = xy(s, "wiper_0")
    # Strike an offset point on the blade from the north to rotate it.
    drive(w + np.array([offset, .52]), 20)
    drive(w + np.array([offset, .05]), 12, cap=.065)
    mid = (xy(s,"wiper_0").copy(), yaw(s), val(s,"wiper_0","z"), base(s).copy())
    # Back away north, center behind the new blade pose, and drive south.
    w = xy(s, "wiper_0")
    drive(w + np.array([0., .58]), 20)
    w = xy(s, "wiper_0")
    drive(np.array([w[0], -.05]), 40, cap=.07)
    final = {n: xy(s, n)-init[n] for n in init}
    print("RESULT", offset, "mid", tuple(np.round(x,3) if not isinstance(x,float) else round(x,3) for x in mid),
          "final_w", np.round(xy(s,"wiper_0"),3), "yaw", round(yaw(s),3),
          "z", round(val(s,"wiper_0","z"),3), "moves", {n:np.round(d,3).tolist() for n,d in final.items()}, flush=True)
    env.close()


if __name__ == "__main__":
    run(float(sys.argv[1]) if len(sys.argv)>1 else -.25,
        int(sys.argv[2]) if len(sys.argv)>2 else 0,
        int(sys.argv[3]) if len(sys.argv)>3 else 5)

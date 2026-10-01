"""Search around an approximate PR2 FK prediction for one blocker grasp."""

import numpy as np

from env_client import make_env


def val(s, name, feature):
    return s.get(s.get_object_from_name(name), feature)


def move_toward(env, s, tx, ty, dlift):
    remaining_lift = dlift
    while (abs(val(s, "robot", "base_x") - tx) > 1e-4 or
           abs(val(s, "robot", "base_y") - ty) > 1e-4 or
           abs(remaining_lift) > 1e-4):
        a = np.zeros(11, dtype=np.float32)
        a[0] = np.clip(tx - val(s, "robot", "base_x"), -0.2, 0.2)
        a[1] = np.clip(ty - val(s, "robot", "base_y"), -0.2, 0.2)
        a[4] = np.clip(remaining_lift, -0.2, 0.2)
        remaining_lift -= a[4]
        old = (val(s, "robot", "base_x"), val(s, "robot", "base_y"))
        s, *_ = env.step(a)
        new = (val(s, "robot", "base_x"), val(s, "robot", "base_y"))
        if new == old and (abs(new[0]-tx) > 1e-4 or abs(new[1]-ty) > 1e-4):
            break
    return s


def main():
    env = make_env()
    try:
        for dlift in np.arange(0.3, 0.901, 0.05):
            s, _ = env.reset(seed=0)
            bx, by = val(s, "blocker", "pose_x"), val(s, "blocker", "pose_y")
            # Approximate FK, with generous raster around predicted base pose.
            cx, cy = bx - (0.732 + 0.15 * dlift), by - (0.417 + 0.06 * dlift)
            s = move_toward(env, s, cx - 0.15, cy - 0.15, dlift)
            for ix, dx in enumerate(np.linspace(-0.15, 0.15, 13)):
                dys = np.linspace(-0.15, 0.15, 13)
                if ix % 2:
                    dys = dys[::-1]
                for dy in dys:
                    s = move_toward(env, s, cx + dx, cy + dy, 0.0)
                    a = np.zeros(11, dtype=np.float32); a[10] = -1.0
                    s, *_ = env.step(a)
                    if val(s, "robot", "grasp_active") > 0.5:
                        print("SUCCESS dlift", dlift, "base", val(s,"robot","base_x"), val(s,"robot","base_y"),
                              "tf", [val(s,"robot",f) for f in ("grasp_tf_x","grasp_tf_y","grasp_tf_z")])
                        return
                    a[10] = 1.0
                    s, *_ = env.step(a)
            print("miss", dlift)
    finally:
        env.close()


if __name__ == "__main__":
    main()

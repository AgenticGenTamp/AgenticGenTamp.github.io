"""Black-box joint/grasp diagnostics for Packing3DEnv."""
from env_client import make_env
import numpy as np


def val(s, o, f):
    return float(s.get(o, f))


def snapshot(s):
    names = s.get_object_names()
    robot = s.get_object_from_name("robot")
    out = {
        "base": [val(s, robot, f"pos_base_{q}") for q in ("x", "y", "rot")],
        "joints": [val(s, robot, f"joint_{i}") for i in range(1, 8)],
        "finger": val(s, robot, "finger_state"),
        "grasp": val(s, robot, "grasp_active"),
        "tf": [val(s, robot, f"grasp_tf_{q}") for q in ("x", "y", "z")],
        "objects": {},
    }
    for name in names:
        if name.startswith("part") or name == "rack":
            o = s.get_object_from_name(name)
            out["objects"][name] = [val(s, o, f"pose_{q}") for q in ("x", "y", "z")]
    return out


def main():
    env = make_env()
    s, info = env.reset(seed=0)
    print("initial", snapshot(s), "info", info)
    # One independent positive/negative pulse per controllable coordinate. Fresh
    # reset avoids contamination and makes this useful for future environments.
    for axis in range(11):
        for sign in (-1, 1):
            s, _ = env.reset(seed=0)
            a = np.zeros(11, dtype=np.float32)
            a[axis] = (1.0 if axis == 10 else 0.2) * sign
            s, r, term, trunc, info = env.step(a)
            print("pulse", axis, sign, snapshot(s), r, term, trunc, info)
    env.close()


def drive_base_and_close(seed=0):
    """Try the default arm pose at each part's lateral coordinate."""
    env = make_env()
    initial, _ = env.reset(seed=seed)
    part_ys = []
    for name in initial.get_object_names():
        if name.startswith("part"):
            o = initial.get_object_from_name(name)
            part_ys.append((name, val(initial, o, "pose_y")))
    rack = initial.get_object_from_name("rack")
    rack_x = val(initial, rack, "pose_x")
    base0_x = val(initial, initial.get_object_from_name("robot"), "pos_base_x")
    for name, target_y in part_ys:
        s, _ = env.reset(seed=seed)
        robot = s.get_object_from_name("robot")
        part = s.get_object_from_name(name)
        # Default gripper is visually above rack center, so preserve the
        # gripper-base offset while translating it over the target part.
        target_x = base0_x + val(s, part, "pose_x") - rack_x
        for _ in range(4):
            dy = target_y - val(s, robot, "pos_base_y")
            dx = target_x - val(s, robot, "pos_base_x")
            if abs(dy) < 1e-5 and abs(dx) < 1e-5:
                break
            a = np.zeros(11, dtype=np.float32)
            a[1] = np.clip(dy, -0.2, 0.2)
            a[0] = np.clip(dx, -0.2, 0.2)
            s, *_ = env.step(a)
        for close in (-1.0, -1.0, 0.0):
            a = np.zeros(11, dtype=np.float32)
            a[10] = close
            s, r, term, trunc, info = env.step(a)
        print("align-close", name, target_y, snapshot(s), r, term, trunc)
    env.close()


if __name__ == "__main__":
    main()
    drive_base_and_close()

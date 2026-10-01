import math
import numpy as np
from env_client import make_env


def val(s, o, f):
    return float(s.get(o, f))


def snapshot(s):
    out = {}
    for name in s.get_object_names():
        o = s.get_object_from_name(name)
        common = ("x", "y", "theta")
        extra = {
            "crv_robot": ("arm_joint", "arm_length", "vacuum", "base_radius",
                          "gripper_width", "gripper_height"),
            "circle": ("radius", "static", "color_r", "color_g", "color_b"),
            "rectangle": ("width", "height", "static", "color_r", "color_g", "color_b"),
        }[o.type.name]
        out[name] = {f: val(s, o, f) for f in common + extra}
    return out


def initial(seed):
    e = make_env()
    s, info = e.reset(seed=seed)
    print("SEED", seed, "info", info, "state", snapshot(s))
    e.close()


def one_step_dynamics(seed=10):
    for action in ([0, 0, 0, .1, 0], [0, 0, .196, 0, 0],
                   [.05, -.05, 0, 0, 1]):
        e = make_env()
        s, _ = e.reset(seed=seed, options={"object_count": 1})
        before = snapshot(s)
        s, r, term, trunc, info = e.step(np.asarray(action, np.float32))
        print("DYN", action, "robot", before["robot"], "->", snapshot(s)["robot"],
              "reward/status/info", r, term, trunc, info)
        e.close()


def drive_base_to_button(seed, vacuum):
    e = make_env()
    s, _ = e.reset(seed=seed, options={"object_count": 1})
    init = snapshot(s)
    bname = next(n for n in s.get_object_names() if n.startswith("button"))
    print("BASE START", seed, "vac", vacuum, init)
    for step in range(100):
        snap = snapshot(s)
        dx = np.clip(snap[bname]["x"] - snap["robot"]["x"], -.05, .05)
        dy = np.clip(snap[bname]["y"] - snap["robot"]["y"], -.05, .05)
        act = np.asarray([dx, dy, 0, -.1, vacuum], np.float32)
        s, r, term, trunc, info = e.step(act)
        ns = snapshot(s)
        dist = math.hypot(ns[bname]["x"] - ns["robot"]["x"],
                          ns[bname]["y"] - ns["robot"]["y"])
        if step % 5 == 0 or term or dist < .2:
            print(" BASE", step + 1, "dist", round(dist, 4), "reward", r,
                  "status", term, trunc, "button", ns[bname], "stick", ns["stick"])
        if term or trunc:
            break
    e.close()


if __name__ == "__main__":
    one_step_dynamics()
    drive_base_to_button(10, 0)
    drive_base_to_button(10, 1)

"""Validate the discovered seed-0 pick/place pose."""
from env_client import make_env
import numpy as np


JOINTS = np.array([0.0, 0.26546195, -np.pi, -2.03142715,
                   0.0, -0.75871509, np.pi / 2])
OFFSET = np.array([0.48035230, 0.00857803])


def g(s, o, f): return float(s.get(o, f))


def move(env, s, robot, base_xy, joints=JOINTS, grip=1.0, steps=5):
    for _ in range(steps):
        a = np.zeros(11, dtype=np.float32)
        a[0] = np.clip(base_xy[0]-g(s, robot, "pos_base_x"), -.2, .2)
        a[1] = np.clip(base_xy[1]-g(s, robot, "pos_base_y"), -.2, .2)
        for i in range(7):
            a[3+i] = np.clip(joints[i]-g(s, robot, f"joint_{i+1}"), -.2, .2)
        a[10] = grip
        s, r, term, trunc, info = env.step(a)
    return s, r, term, trunc


def main():
    env = make_env(); s, _ = env.reset(seed=0)
    robot = s.get_object_from_name("robot")
    rack = s.get_object_from_name("rack")
    rack_xy = np.array([g(s, rack, "pose_x"), g(s, rack, "pose_y")])
    for name in ("part0", "part1"):
        part = s.get_object_from_name(name)
        part_xy = np.array([g(s, part, "pose_x"), g(s, part, "pose_y")])
        correction = np.array([0.0, 0.0]) if name == "part0" else np.array([0.03, 0.03])
        pick_base = part_xy-OFFSET+correction
        s, *_ = move(env, s, robot, pick_base, grip=1.0)
        s, r, term, trunc = move(env, s, robot, pick_base, grip=-1.0, steps=1)
        print(name, "picked", g(s, robot, "grasp_active"), g(s, part, "grasp_active"),
              [g(s, part, f"pose_{q}") for q in "xyz"])
        # Joint 2 negative cleanly lifts the held part about 5.6 cm.
        lifted = JOINTS.copy(); lifted[1] -= 0.15
        s, *_ = move(env, s, robot, pick_base, joints=lifted,
                     grip=0.0, steps=1)
        lift_dx = 0.0235403
        # Slight y separation allows multiple 10cm-wide parts in 30cm rack.
        # Put each part on the same side it approaches from; crossing over an
        # already placed part can block the held object's base translation.
        slot_y = (0.07 if name == "part0" else -0.07)
        place_base = rack_xy + [0, slot_y] - OFFSET + correction - [lift_dx, 0]
        s, *_ = move(env, s, robot, place_base, joints=lifted,
                     grip=0.0, steps=5)
        # Descend in small increments: a single 0.15-rad command is rejected
        # wholesale when its endpoint would penetrate the rack.
        descent = ([0.02] * 8 if name == "part0" else
                   [0.02, 0.02, 0.005, 0.005, 0.005, 0.001, 0.001, 0.001])
        for increment in descent:
            a = np.zeros(11, dtype=np.float32); a[4] = increment
            s, *_ = env.step(a)
        if name == "part1":
            # A small elbow push while opening breaks the triangle grasp once
            # its lower face is in contact with the rack.
            a = np.zeros(11, dtype=np.float32); a[6] = -0.02; a[10] = 1.0
            s, *_ = env.step(a)
        for _ in range(7):
            a = np.zeros(11, dtype=np.float32); a[10] = 1.0
            s, r, term, trunc, _ = env.step(a)
        print(name, "placed", g(s, robot, "grasp_active"), g(s, part, "grasp_active"),
              [g(s, part, f"pose_{q}") for q in "xyz"],
              "finger", g(s, robot, "finger_state"),
              "q2", g(s, robot, "joint_2"), r, term, trunc)
    env.close()


if __name__ == "__main__": main()

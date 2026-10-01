"""After the known grasp, test which joint pulse lifts/moves the part."""
from env_client import make_env
from probe_pick_place import JOINTS, OFFSET, g, move
import numpy as np


def run(axis, sign):
    env = make_env(); s, _ = env.reset(seed=0)
    robot = s.get_object_from_name("robot"); part = s.get_object_from_name("part0")
    xy = np.array([g(s, part, "pose_x"), g(s, part, "pose_y")])
    s, *_ = move(env, s, robot, xy-OFFSET, grip=1.0)
    s, *_ = move(env, s, robot, xy-OFFSET, grip=-1.0, steps=1)
    before = np.array([g(s, part, f"pose_{q}") for q in "xyz"])
    a = np.zeros(11, dtype=np.float32); a[axis] = .15*sign; a[10] = 0
    s, *_ = env.step(a)
    after = np.array([g(s, part, f"pose_{q}") for q in "xyz"])
    print(axis, sign, "delta", after-before, "grasp", g(s, part, "grasp_active"))
    env.close()


if __name__ == "__main__":
    for axis in range(3, 10):
        for sign in (-1, 1): run(axis, sign)

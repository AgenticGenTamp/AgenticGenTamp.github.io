"""Find shape-dependent XY correction for the triangular part."""
from env_client import make_env
from probe_pick_place import JOINTS, OFFSET, g, move
import numpy as np


def main(seed=1):
    env = make_env(); s, _ = env.reset(seed=seed, options={"object_count": 1})
    robot = s.get_object_from_name("robot"); part = s.get_object_from_name("part0")
    xy = np.array([g(s, part, "pose_x"), g(s, part, "pose_y")])
    # Interior-biased offsets for either diagonal orientation.
    corrections = [(dx, dy) for dx in (0, -.015, .015, -.03, .03, -.045, .045)
                   for dy in (0, -.015, .015, -.03, .03, -.045, .045)]
    for dx, dy in corrections:
        base = xy - OFFSET + [dx, dy]
        s, *_ = move(env, s, robot, base, grip=1.0)
        s, *_ = move(env, s, robot, base, grip=-1.0, steps=1)
        if g(s, part, "grasp_active"):
            print("SUCCESS correction", dx, dy, "base", base.tolist())
            env.close(); return
    print("none"); env.close()


if __name__ == "__main__": main()

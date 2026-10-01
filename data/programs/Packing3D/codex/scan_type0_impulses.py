"""Scan one deterministic impulse at the common type-0 rack contact pose."""
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach


def g(s, name, feature):
    return float(s.get(s.get_object_from_name(name), feature))


def run(code, seed=1):
    env = make_env(); s, info = env.reset(seed=seed, options={"object_count": 1})
    policy = GeneratedApproach(env.action_space, env.observation_space, env.make_primitives())
    policy.reset(s, info)
    # Stop just as the stock policy has finished pressing downward, before release.
    for step in range(80):
        action = policy.get_action(s)
        if policy.phase == "release":
            break
        s, _, term, trunc, _ = env.step(action)
    print("pre", step, policy.phase, g(s,"robot","grasp_active"),
          [round(g(s,"part0",f),5) for f in ("pose_x","pose_y","pose_z")],
          [round(g(s,"robot",f),5) for f in ("joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7")])
    axis, amount, repeats = code.split(":")
    axis = int(axis); amount = float(amount); repeats = int(repeats)
    term = False
    for k in range(repeats):
        a = np.zeros(11, np.float32); a[axis] = amount; a[10] = 1.
        s, _, term, trunc, _ = env.step(a)
        print("post", k+1, "hold",g(s,"robot","grasp_active"),
              "part",[round(g(s,"part0",f),5) for f in ("pose_x","pose_y","pose_z")],
              "robot",[round(g(s,"robot",f),5) for f in ("pos_base_x","pos_base_y","pos_base_rot","joint_2","joint_4")],
              "term",term)
        if term or trunc: break
    env.close()
    return term


if __name__ == "__main__": run(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else 1)

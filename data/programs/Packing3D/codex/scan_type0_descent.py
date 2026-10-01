"""Find the minimum number of 0.02 joint-2 seating pulses before release."""
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach


def g(s, name, feature):
    return float(s.get(s.get_object_from_name(name), feature))


def run(pulses, seed=1):
    env = make_env(); s, info = env.reset(seed=seed, options={"object_count": 1})
    policy = GeneratedApproach(env.action_space, env.observation_space, env.make_primitives())
    policy.reset(s, info); descents = 0
    for step in range(40):
        action = policy.get_action(s)
        if policy.phase == "descend": descents += 1
        if descents > pulses:
            action = np.zeros(11, np.float32); action[6] = -.02; action[10] = 1.
        s, _, term, trunc, _ = env.step(action)
        if descents > pulses or term or trunc: break
    print("pulses",pulses,"seed",seed,"steps",step+1,
          "hold",g(s,"robot","grasp_active"),
          "part",[round(g(s,"part0",f),5) for f in ("pose_x","pose_y","pose_z")],
          "j2",round(g(s,"robot","joint_2"),5),"term",term)
    env.close()


if __name__ == "__main__": run(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv)>2 else 1)

"""Log robot configuration and object motion around paddle saturation."""

import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def main():
    for seed in (0, 1, 3):
        env = make_env()
        state, info = env.reset(seed=seed)
        policy = GeneratedApproach(env.action_space, env.observation_space, {})
        policy.reset(state, info)
        for step in range(1, 341):
            state, reward, term, trunc, _ = env.step(policy.get_action(state))
            if step in (190, 200, 210, 220, 230, 240, 250, 260, 270, 300, 330):
                rob = state.get_object_from_name("robot")
                joints = [state.get(rob, "pos_arm_joint%d" % i)
                          for i in range(1, 8)]
                base = [state.get(rob, f) for f in
                        ("pos_base_x", "pos_base_y", "pos_base_rot")]
                cubes = [state.get_object_from_name(n)
                         for n in state.get_object_names()
                         if n.startswith("cube_")]
                cy = np.mean([state.get(c, "y") for c in cubes])
                print(seed, step, "base", np.round(base, 3),
                      "joints", np.round(joints, 3), "cube_y", round(cy, 3))
            if term or trunc:
                break
        env.close()


if __name__ == "__main__":
    main()

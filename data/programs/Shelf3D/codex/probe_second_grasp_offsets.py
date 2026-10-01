"""Try calibrated base offsets for the second grasp without changing policy."""

import sys
import numpy as np

from env_client import make_env
from probe_baseline_loader import GeneratedApproach


def main(dx, dy):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 2})
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, {})
    highest = 0.0
    try:
        for step in range(920):
            action = policy.get_action(state)
            # Offset all second-pick base targets consistently.  During stage
            # zero preserve the policy's 19.5 cm retreat.
            if policy.index == 1 and policy.stage <= 5 and policy.pick_xy is not None:
                rob = state.get_object_from_name("robot")
                bx = float(state.get(rob, "pos_base_x"))
                by = float(state.get(rob, "pos_base_y"))
                tx = policy.pick_xy[0] - .609 + dx
                ty = policy.pick_xy[1] - .054 + dy
                if policy.stage == 0:
                    tx -= .195
                action[0] = np.clip((tx - bx) / .87, -.1, .1)
                action[1] = np.clip((ty - by) / .87, -.1, .1)
            state, _, term, trunc, _ = env.step(action)
            cube = state.get_object_from_name("cube2")
            p = tuple(float(state.get(cube, f)) for f in ("x", "y", "z"))
            highest = max(highest, p[2])
            if term or trunc:
                break
        print("offset", dx, dy, "highest", round(highest, 4),
              "final", tuple(round(v, 4) for v in p),
              "policy", policy.index, policy.stage, policy.retry)
    finally:
        env.close()


if __name__ == "__main__":
    main(float(sys.argv[1]), float(sys.argv[2]))

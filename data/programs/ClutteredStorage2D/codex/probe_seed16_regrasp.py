"""Test alternate released-block regrasp routes on cluttered seed 16."""

import sys
import math

from approach import GeneratedApproach
from env_client import make_env


seed = int(sys.argv[1]) if len(sys.argv) > 1 else 16
mode = sys.argv[2] if len(sys.argv) > 2 else "smart"
env = make_env()
state, info = env.reset(seed=seed)
policy = GeneratedApproach(env.action_space, env.observation_space,
                           env.make_primitives())
policy.reset(state, info)
last_stage = None

for step in range(1000):
    action = policy.get_action(state)
    if policy.stage == "reposition_side" and mode in ("right", "smart"):
        block = state.get_object_from_name(policy.target_name)
        bx, by = (float(state.get(block, f)) for f in ("x", "y"))
        side = 1.0
        if mode == "smart":
            others = [b for b in state.get_objects(policy.block_type)
                      if b.name != policy.target_name and
                      float(state.get(b, "y")) < 2.4]
            def clearance(sign):
                x = bx + sign * .54
                edge = min(x-.22, 4.78-x)
                return min([edge] + [abs(x-float(state.get(b, "x")))-.30
                                     for b in others])
            scores = {s: clearance(s) for s in (-1.0, 1.0)}
            side = max(scores, key=lambda s: (scores[s],
                       -abs((bx+s*.54)-float(state.get(
                           state.get_object_from_name("robot"), "x")))))
        policy.regrasp_side_x = min(4.7, max(.3, bx + side * 0.60))
        policy.approach_theta = math.pi if side > 0 else 0.0
        policy.goal_base = (bx + side * 0.54, by)
        policy.regrasp_safe = (policy.goal_base[0], max(0.3, by - 0.85))
    elif policy.stage == "reposition_side" and mode == "wide_left":
        block = state.get_object_from_name(policy.target_name)
        bx = float(state.get(block, "x"))
        policy.regrasp_side_x = max(0.3, bx - 0.80)
    if policy.stage != last_stage:
        robot = state.get_object_from_name("robot")
        block = state.get_object_from_name(policy.target_name) if policy.target_name else None
        print(step, policy.stage,
              tuple(round(float(state.get(robot, f)), 3)
                    for f in ("x", "y", "theta", "arm_joint")),
              None if block is None else tuple(round(float(state.get(block, f)), 3)
                                                for f in ("x", "y", "theta")))
        last_stage = policy.stage
    state, _, terminated, truncated, step_info = env.step(action)
    if terminated or truncated:
        print("RESULT", step + 1, terminated, truncated, step_info)
        break
else:
    print("RESULT", 1000, False, False, step_info)

print("blocks", [(b.name,) + tuple(round(float(state.get(b, f)), 3)
                                    for f in ("x", "y", "theta"))
                 for b in state.get_objects(policy.block_type)])
env.close()

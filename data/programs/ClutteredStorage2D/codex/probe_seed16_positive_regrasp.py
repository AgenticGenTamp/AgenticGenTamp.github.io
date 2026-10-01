"""Relocate a reversed first payload before its end-on regrasp."""

import sys

from approach import GeneratedApproach
from env_client import make_env


seed = int(sys.argv[1]) if len(sys.argv) > 1 else 16
env = make_env()
state, info = env.reset(seed=seed)
policy = GeneratedApproach(env.action_space, env.observation_space,
                           env.make_primitives())
policy.reset(state, info)
relocated = set()
relocating = set()
relocation_goal = {}
relocation_low = set()
last_stage = None

for step in range(1000):
    action = policy.get_action(state)
    # Intercept the policy's own relocation before its first horizontal step.
    if policy.stage == "orient" and getattr(policy, "relocating", False):
        block = state.get_object_from_name(policy.target_name)
        bx, by = (float(state.get(block, f)) for f in ("x", "y"))
        goal_x = getattr(policy, "relocation_x", 2.45)
        lo, hi = sorted((bx, goal_x))
        others = [q for q in policy._blocks(state)
                  if q.name != policy.target_name and not policy._inside(state, q)]
        if any(lo-.05 <= float(state.get(q, "x")) <= hi+.05 and
               abs(float(state.get(q, "y"))-by) < .45 for q in others):
            relocation_low.add(policy.target_name)
        if policy.target_name in relocation_low and by < 2.09:
            action = policy._action(dy=2.10-by, vac=1.0)
        if seed == 4 and policy.target_name == "block5":
            print("reloc", step, round(bx, 3), round(by, 3), round(goal_x, 3),
                  policy.target_name in relocation_low,
                  [(q.name, round(float(state.get(q, "x")), 3),
                    round(float(state.get(q, "y")), 3)) for q in others])
    if policy.stage == "drop_for_regrasp" and policy.target_name not in relocated:
        block = state.get_object_from_name(policy.target_name)
        bx = float(state.get(block, "x"))
        robot = state.get_object_from_name("robot")
        rx, ry = (float(state.get(robot, f)) for f in ("x", "y"))
        side = 1.0 if rx >= bx else -1.0
        trial_x = bx + .50 * side
        others = [q for q in policy._blocks(state)
                  if q.name != policy.target_name and not policy._inside(state, q)]
        blocked = any(((trial_x-float(state.get(q, "x")))**2 +
                       (ry-float(state.get(q, "y")))**2)**.5 < .28
                      for q in others)
        left_goal = bx - .54
        path_lo, path_hi = sorted((rx, left_goal))
        low_route_blocked = any(path_lo-.1 <= float(state.get(q, "x")) <= path_hi+.1
                                and abs(float(state.get(q, "y"))-(float(
                                    state.get(block, "y"))-.50)) < .50
                                for q in others)
        would_grasp_right = ((blocked and -side > 0) or bx-.54 < .22 or
                             low_route_blocked)
        if would_grasp_right:
            relocating.add(policy.target_name)
            relocation_goal[policy.target_name] = 2.45
            lo, hi = sorted((bx, 2.45))
            if any(lo-.50 <= float(state.get(q, "x")) <= hi+.10 and
                   abs(float(state.get(q, "y"))-float(state.get(block, "y"))) < .32
                   for q in others):
                relocation_low.add(policy.target_name)
        # Keep the horizontal payload attached and shift it away from nearby
        # loose blocks.  Re-enter orient until the shift is complete.
        goal_x = relocation_goal.get(policy.target_name, 2.45)
        by = float(state.get(block, "y"))
        if policy.target_name in relocating and policy.target_name in relocation_low and by < 2.09:
            policy.stage = "orient"
            action = policy._action(dy=2.10-by, vac=1.0)
        elif policy.target_name in relocating and abs(bx-goal_x) >= .01:
            policy.stage = "orient"
            action = policy._action(dx=goal_x-bx, vac=1.0)
        else:
            relocated.add(policy.target_name)
    if policy.stage != last_stage:
        block = (state.get_object_from_name(policy.target_name)
                 if policy.target_name else None)
        print(step, policy.stage, policy.target_name,
              None if block is None else tuple(round(float(state.get(block, f)), 3)
                                                for f in ("x", "y", "theta")))
        last_stage = policy.stage
    state, _, terminated, truncated, step_info = env.step(action)
    if terminated or truncated:
        print("RESULT", step+1, terminated, truncated, step_info)
        break
else:
    print("RESULT", 1000, False, False, step_info)

print("blocks", [(b.name,) + tuple(round(float(state.get(b, f)), 3)
                                    for f in ("x", "y", "theta"))
                 for b in state.get_objects(policy.block_type)])
env.close()

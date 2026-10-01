"""Probe small final-slot offsets for seed 16's tightly packed shelf."""

import sys

from approach import GeneratedApproach
from env_client import make_env


seed = int(sys.argv[1]) if len(sys.argv) > 1 else 16
offset = float(sys.argv[2]) if len(sys.argv) > 2 else -0.01
mode = sys.argv[3] if len(sys.argv) > 3 else "nudge"
env = make_env()
state, info = env.reset(seed=seed)
policy = GeneratedApproach(env.action_space, env.observation_space,
                           env.make_primitives())
policy.reset(state, info)
adjusted = set()

for step in range(1000):
    action = policy.get_action(state)
    # Apply once after choose has set this target's slot and preplace.
    if mode == "slot" and policy.target_name and policy.target_name not in adjusted:
        inside = sum(policy._inside(state, b) for b in policy._blocks(state))
        if inside >= 2:
            old = policy.place
            policy.place = (old[0] + offset, old[1])
            policy.preplace = (policy.preplace[0] + offset, policy.preplace[1])
            adjusted.add(policy.target_name)
            print("adjust", step, policy.target_name, old, "to", policy.place)
    if mode == "raise" and policy.target_name and policy.target_name not in adjusted:
        others = sum(b.name != policy.target_name and policy._inside(state, b)
                     for b in policy._blocks(state))
        if others < 2:
            old = policy.place
            policy.place = (old[0], old[1] + offset)
            adjusted.add(policy.target_name)
            print("raise", step, policy.target_name, old, "to", policy.place)
    if (mode == "firstedge" and not policy.clear_mode and policy.target_name and
            policy.target_name not in adjusted):
        others = sum(b.name != policy.target_name and policy._inside(state, b)
                     for b in policy._blocks(state))
        if others == 0:
            shelf = state.get_object_from_name("shelf")
            edge = (float(state.get(shelf, "x1")) +
                    float(state.get(shelf, "width1")) - .071)
            old = policy.place
            policy.place = (edge, old[1])
            policy.preplace = (edge, policy.preplace[1])
            adjusted.add(policy.target_name)
            print("firstedge", step, policy.target_name, old, "to", policy.place)
    if mode == "nudge" and policy.stage == "insert" and policy.target_name:
        block = state.get_object_from_name(policy.target_name)
        bx, by = (float(state.get(block, f)) for f in ("x", "y"))
        other_stored = sum(b.name != policy.target_name and policy._inside(state, b)
                           for b in policy._blocks(state))
        if other_stored >= 2 and by > 2.28 and policy.insert_toggle == 0:
            action[0] = max(-.05, min(.05, policy.place[0] + offset - bx))
    state, _, terminated, truncated, step_info = env.step(action)
    if step % 100 == 99:
        block = (state.get_object_from_name(policy.target_name)
                 if policy.target_name else None)
        print(step+1, policy.stage,
              None if block is None else tuple(round(float(state.get(block, f)), 3)
                                                for f in ("x", "y", "theta")))
    if terminated or truncated:
        print("RESULT", step+1, terminated, truncated, step_info)
        break
else:
    print("RESULT", 1000, False, False, step_info)

print("blocks", [(b.name,) + tuple(round(float(state.get(b, f)), 3)
                                    for f in ("x", "y", "theta"))
                 for b in state.get_objects(policy.block_type)])
env.close()

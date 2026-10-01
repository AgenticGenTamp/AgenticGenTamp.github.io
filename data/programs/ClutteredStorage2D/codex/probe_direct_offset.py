"""Probe tangential offsets for seed-0 direct pickup of block6."""
import math
import sys
from env_client import make_env
from approach import GeneratedApproach

offset = float(sys.argv[1])
env = make_env()
state, info = env.reset(seed=0)
p = GeneratedApproach(env.action_space, env.observation_space, env.make_primitives())
p.reset(state, info)
adjusted = False
for step in range(1000):
    action = p.get_action(state)
    if (p.target_name == "block6" and p.direct_pick and p.stage == "navigate"
            and not adjusted):
        th = p.approach_theta
        p.goal_base = (p.goal_base[0] - offset * math.sin(th),
                       p.goal_base[1] + offset * math.cos(th))
        adjusted = True
    state, _, term, trunc, _ = env.step(action)
    if term or trunc:
        break
    if adjusted and p.target_name == "block6" and p.stage in ("orient", "lower_transport"):
        break
b = state.get_object_from_name("block6")
print(offset, step + 1, p.stage,
      tuple(round(float(state.get(b, q)), 3) for q in ("x", "y", "theta")))
env.close()

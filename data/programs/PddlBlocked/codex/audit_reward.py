"""Read-only route/step profiler for the current GeneratedApproach."""
from collections import defaultdict
import math
import sys

from env_client import make_env
from approach import GeneratedApproach

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
stop = int(sys.argv[2]) if len(sys.argv) > 2 else start + 120
env = make_env()
rows = []
stage_totals = defaultdict(list)
for seed in range(start, stop):
    state, info = env.reset(seed=seed)
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    initial = policy.stage
    counts = defaultdict(int)
    terminated = truncated = False
    for step in range(env.max_steps):
        stage = policy.stage
        state, _, terminated, truncated, _ = env.step(policy.get_action(state))
        counts[stage] += 1
        if terminated or truncated:
            break
    angle = math.degrees(math.atan2(policy.out[1], policy.out[0]))
    mode = "spare" if initial == -5 else "local"
    if mode == "spare":
        geom = "spare"
    else:
        geom = ("E" if -45 <= angle < 45 else "N" if 45 <= angle < 135
                else "W" if angle >= 135 or angle < -135 else "S")
    rows.append((seed, terminated, step + 1, mode, geom, angle, dict(counts)))
    if terminated:
        stage_totals[(mode, geom)].append(step + 1)

for key, values in sorted(stage_totals.items()):
    values.sort()
    print(key, "n", len(values), "min/med/max", values[0], values[len(values)//2], values[-1],
          "mean", round(sum(values)/len(values), 1))
print("slow successes")
for row in sorted((r for r in rows if r[1]), key=lambda r: r[2], reverse=True)[:15]:
    print(row)
print("failures", [(r[0], r[4], round(r[5], 1)) for r in rows if not r[1]])
env.close()

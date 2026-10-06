"""Categorise initial states: wall-blocked pusher, insufficient room ahead, etc."""
import sys
from collections import Counter
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach, WORLD_MAX_X

pairs = [tuple(map(int, p.split(":"))) for p in sys.argv[1].split(",")]
counts = sorted({p[0] for p in pairs})
env = VariableObjectCountEnv(
    constant_object_env_path="kinder.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv",
    count_kwarg="num_obstructions", count_object_prefix="obstruction",
    design_counts=counts, eval_counts=counts)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
cat = Counter()
for n, seed in pairs:
    if True:
        try:
            state, _ = env.reset(seed=seed, options={"object_count": n})
        except RuntimeError:
            continue
        r, b, s, obs = ap._parse(state)
        plan = ap._make_plan(r, b, s, obs)
        d = plan["d"]
        names = {o["name"] for o in plan["chain"]}
        ahead = [o for o in obs if (o["x"] - b["x"]) * d > 0]
        wsum = sum(o["width"] for o in ahead)
        sl, sr = s["x"] - s["width"] / 2, s["x"] + s["width"] / 2
        room = (WORLD_MAX_X - (sl + b["width"])) if d > 0 else (sr - b["width"])
        tags = []
        if plan["wall"]:
            tags.append("wall")
        if wsum > room - 0.01:
            tags.append("noroom")
        if b["width"] < 0.24:
            tags.append("narrowT")
        print(n, seed, tags, 'd', d, 'T', round(b['x'],2), round(b['width'],2), 'S', round(s['x'],2), round(s['width'],2), 'ahead', [(round(o['x'],2), round(o['width'],2)) for o in ahead], 'room', round(room,2))
for k in sorted(cat):
    print(k, cat[k])

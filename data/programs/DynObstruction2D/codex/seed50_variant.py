"""Focused A/B: fully clear the narrow cargo corridor before sweeping."""
import sys
import numpy as np
from env_client import make_env

source = open("approach.py", encoding="utf-8").read()
source = source.replace(
    "if self._overlap_depth(ox, ow, sx, sw) > .04:\n"
    "                        candidates.append((abs(ox - sx), name, o))",
    "if self._overlap_depth(ox, ow, sx, sw + 2.0 * bw) > .04:\n"
    "                        candidates.append((abs(ox - sx), name, o))")
source = source.replace(
    "if sw > .55 or (len(candidates) >= 2 and\n"
    "                            min(g(item[2], \"height\") for item in candidates) < .30):",
    "if sw > .55:")
source = source.replace(
    "goal = sx + self.direction * (sw / 2.0 + ow / 2.0 + .10)",
    "goal = sx + self.direction * (sw / 2.0 + ow / 2.0 + .10 + bw)")
source = source.replace(
    "if not candidates:\n"
    "                self.direction = 1 if sx >= bx else -1\n"
    "                self._advance(\"raise\")",
    "if not candidates:\n"
    "                if bw < .22 and abs(sx - bx) > .60 and not pad_blocked_now:\n"
    "                    self.boundary_pick = True\n"
    "                    self._advance(\"boundary_raise\")\n"
    "                else:\n"
    "                    self.direction = 1 if sx >= bx else -1\n"
    "                    self._advance(\"raise\")")
namespace = {}
exec(compile(source, "seed50_variant", "exec"), namespace)
Policy = namespace["GeneratedApproach"]

for seed in map(int, sys.argv[1:]):
    env = make_env(); state, info = env.reset(seed=seed)
    policy = Policy(env.action_space, env.observation_space, {})
    policy.reset(state, info); term = trunc = False
    for step in range(env.max_steps):
        state, _, term, trunc, info = env.step(np.asarray(
            policy.get_action(state), dtype=env.action_space.dtype))
        if term or trunc: break
    block = state.get_objects(env.observation_space.get_type("target_block"))[0]
    print(seed, bool(term), step + 1, policy.phase,
          round(float(state.get(block, "x")), 3),
          round(float(state.get(block, "y")), 3), flush=True)
    env.close()

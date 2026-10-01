"""Sweep a seed interval and report compact PR2Blocked failure diagnostics."""
import math
import sys

import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def f(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


lo = int(sys.argv[1]) if len(sys.argv) > 1 else 200
hi = int(sys.argv[2]) if len(sys.argv) > 2 else 500
env = make_env()
wins = 0
steps_sum = 0
failures = []
for seed in range(lo, hi):
    state, info = env.reset(seed=seed)
    names = state.get_object_names()
    spares = [n for n in names if n.startswith("green") and n != "green0"]
    green = np.array([f(state, "green0", "pose_x"), f(state, "green0", "pose_y")])
    blocker = np.array([f(state, "blocker", "pose_x"), f(state, "blocker", "pose_y")])
    delta = blocker - green
    delta /= np.linalg.norm(delta)
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    mode = "spare" if policy.stage == -5 else ("west" if policy.west_branch else "axis")
    stage_visits = {}
    rejected = 0
    last_stage = policy.stage
    for step in range(env.max_steps):
        before_base = policy.robot(state)
        before_joints = np.array([f(state, "robot", "joint_" + str(j)) for j in range(1, 8)])
        action = policy.get_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        after_base = policy.robot(state)
        after_joints = np.array([f(state, "robot", "joint_" + str(j)) for j in range(1, 8)])
        stage_visits[policy.stage] = stage_visits.get(policy.stage, 0) + 1
        commanded = np.max(np.abs(action[:10])) > 1e-5
        moved = max(np.max(np.abs(after_base - before_base)), np.max(np.abs(after_joints - before_joints)))
        if commanded and moved < 1e-7:
            rejected += 1
        last_stage = policy.stage
        if terminated or truncated:
            break
    nsteps = step + 1
    if terminated:
        wins += 1
        steps_sum += nsteps
    else:
        hot = max(stage_visits, key=stage_visits.get)
        failures.append((seed, len(spares), mode, delta[0], delta[1], green[0], green[1],
                         blocker[0], blocker[1], last_stage, hot, stage_visits[hot], rejected,
                         f(state, "robot", "grasp_active"), policy.robot(state)[0], policy.robot(state)[1]))

for x in failures:
    print("FAIL seed=%d nsp=%d mode=%s out=(%+.3f,%+.3f) g=(%.3f,%.3f) b=(%.3f,%.3f) "
          "stage=%d hot=%d:%d rejected=%d held=%.0f base=(%.3f,%.3f)" % x)
print("SUMMARY range=%d:%d wins=%d/%d mean_win_steps=%.1f failures=%d" %
      (lo, hi, wins, hi - lo, steps_sum / wins if wins else math.nan, len(failures)))
env.close()

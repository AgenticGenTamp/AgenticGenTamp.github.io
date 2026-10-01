"""Tip red tray on its side, then push cube1 through its measured opening."""
import numpy as np

from env_client import make_env
from edge_rake_probe import action_to, robot
from topdown_grasp_probe import get, xyz, act, run
from probe_tip_bin import HOME, HIGH, LOW, PUSH, opening_axis, pose


def sweep(env, state, unit, stop):
    """Calibrated positive-q1 fingertip sweep in an arbitrary world direction."""
    unit = np.asarray(unit, dtype=float)
    unit /= np.linalg.norm(unit)
    yaw = float(np.arctan2(unit[0], -unit[1]))
    cube = xyz(state, "cube1").copy()
    c, s = np.cos(yaw), np.sin(yaw)

    def offset(lx, ly):
        return np.array([cube[0] + c * lx - s * ly,
                         cube[1] + s * lx + c * ly, yaw])

    outer, contact = offset(-.97, .042), offset(-.75, .042)
    # Fold, move to the outer perimeter at the destination polar angle, then
    # deploy the narrow fingertip without crossing the tabletop.
    current = robot(state)[:3].copy()
    state = run(env, state, 25, current, HOME, 1)
    start_angle = float(np.arctan2(current[1], current[0]))
    end_angle = float(np.arctan2(outer[1], outer[0]))
    turn = (end_angle - start_angle + np.pi) % (2 * np.pi) - np.pi
    for k in range(40):
        angle = start_angle + turn * ((k + 1) / 40.)
        state = run(env, state, 1,
                    [1.15 * np.cos(angle), 1.15 * np.sin(angle), yaw], HOME, 1)
    state = run(env, state, 25, outer, HOME, 1)
    state = run(env, state, 90, outer, PUSH, 1)

    before = xyz(state, "cube1").copy()
    hit = None
    for _ in range(45):
        action = action_to(state, contact, PUSH, 1)
        action[:3] = np.clip(action[:3], -.012, .012)
        state, reward, term, trunc, _ = env.step(action)
        if np.linalg.norm(xyz(state, "cube1") - before) > .002:
            hit = robot(state)[:3].copy()
            break
    if hit is None:
        return state, reward, term, "no_hit"
    qgoal = PUSH.copy(); qgoal[0] = .82
    for _ in range(100):
        state, reward, term, trunc, _ = env.step(action_to(state, hit, qgoal, 1))
        if stop(state) or term or trunc:
            break
    return state, reward, term, "hit"


env = make_env()
state, _ = env.reset(seed=0, options={"object_count": 4})
b0, c0 = xyz(state, "bin_red").copy(), xyz(state, "cube1").copy()

# Reproduce the reliable q4-positive 90-degree tip, without the later bin push.
state = run(env, state, 45, [1., .75, np.pi], HOME, 1)
state = run(env, state, 55, [-1.1, .75, np.pi], HOME, 1)
state = run(env, state, 45, [-1.1, b0[1], 0.], HOME, 1)
state = run(env, state, 110, [-1.1, b0[1], 0.], HIGH, 1)
base = np.array([b0[0] - .91, b0[1], 0.])
state = run(env, state, 30, base, HIGH, 1)
state = run(env, state, 45, base, LOW, 1)
state = run(env, state, 25, base, LOW, 0)
direction = c0[:2] - b0[:2]; direction /= np.linalg.norm(direction)
tip_q = LOW.copy(); tip_q[1] = .88; tip_q[3] += .9
goal = base.copy()
for k in range(12):
    goal[:2] += .012 * direction
    action = act(state, goal, tip_q, 0 if k < 7 else 1)
    action[:2] = np.clip(action[:2], -.05, .05)
    action[3:10] = np.clip(action[3:10], -.06, .06)
    state, reward, term, trunc, _ = env.step(action)
state = run(env, state, 90, goal, tip_q, 1)

bp = pose(state, "bin_red")
axis = opening_axis(bp[3:])
axis[:2] /= np.linalg.norm(axis[:2])
cube = xyz(state, "cube1")
# Opening centerline is bin_xy + t*axis. Remove cube's perpendicular error.
relative = cube[:2] - bp[:2]
perp = relative - np.dot(relative, axis[:2]) * axis[:2]
transverse_unit = -perp / max(1e-8, np.linalg.norm(perp))
line_tolerance = .012
state, reward, term, status1 = sweep(
    env, state, transverse_unit,
    lambda st: abs(np.cross(axis[:2], xyz(st, "cube1")[:2] -
                            pose(st, "bin_red")[:2])) < line_tolerance)

# Re-measure because the transverse sweep may slightly rotate the tray.
bp = pose(state, "bin_red")
axis = opening_axis(bp[3:]); axis[:2] /= max(1e-8, np.linalg.norm(axis[:2]))
# Move opposite the outward opening axis, i.e. into the cavity.
state, reward, term, status2 = sweep(
    env, state, -axis[:2],
    lambda st: np.linalg.norm(xyz(st, "cube1")[:2] -
                              pose(st, "bin_red")[:2]) < .025)

state = run(env, state, 80, robot(state)[:3], HOME, 1)
bp, cp = pose(state, "bin_red"), pose(state, "cube1")
print("PUSH_IN", status1, status2,
      "bin", np.round(bp, 5).tolist(), "cube", np.round(cp, 5).tolist(),
      "relative", np.round(cp[:3] - bp[:3], 5).tolist(),
      "dxy", round(float(np.linalg.norm(cp[:2] - bp[:2])), 5),
      "reward", reward, "term", term)
env.close()

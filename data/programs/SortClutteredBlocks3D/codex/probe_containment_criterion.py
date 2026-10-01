"""Test whether close XY alignment or genuine tray entry earns credit.

This repeats the calibrated seed-0 red-cube sweep, records the complete pose at
the closest approach to the red tray's *initial* center, and optionally raises
the contacting fingertip while gently biasing the cube toward that center.
"""
import sys

import numpy as np

from env_client import make_env
from edge_rake_probe import Q, action_to, cubes, get, robot, settle_to
from approach import GeneratedApproach


def pose(state, name):
    return get(state, name, ["x", "y", "z", "qw", "qx", "qy", "qz"])


def run(mode):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 4})
    initial = cubes(state)
    name = min(initial, key=lambda n: initial[n][1])
    target = get(state, "bin_red", ["x", "y", "z"]).copy()

    state = settle_to(env, state, [.99, .80, np.pi], None, 45, 0)
    state = settle_to(env, state, [-1., .80, np.pi], None, 55, 0)
    state = settle_to(env, state, [-1., .80, 0.], None, 45, 0)
    ybase = float(initial[name][1] + .042)
    state = settle_to(env, state, [-1., ybase, 0.], None, 45, 0)
    state = settle_to(env, state, [-1., ybase, 0.], Q, 130, 0)

    hit = None
    for _ in range(50):
        action = action_to(state, [-.80, ybase, 0.], Q, 0)
        action[0] = min(action[0], .012)
        state, reward, term, trunc, _ = env.step(action)
        if max(np.linalg.norm(v - initial[n]) for n, v in cubes(state).items()) > .001:
            hit = robot(state)[:3].copy()
            break
    if hit is None:
        print(mode, "NO_CONTACT")
        env.close()
        return

    qgoal = Q.copy()
    qgoal[0] = .82
    base_goal = hit.copy()
    best = None
    for step in range(150):
        cube = cubes(state)[name]
        base_goal[1] += float(np.clip(1.4 * (target[1] - cube[1]), -.012, .012))
        state, reward, term, trunc, _ = env.step(action_to(state, base_goal, qgoal, 0))
        cp = pose(state, name)
        bp = pose(state, "bin_red")
        d_fixed = float(np.linalg.norm(cp[:2] - target[:2]))
        d_live = float(np.linalg.norm(cp[:2] - bp[:2]))
        record = (d_fixed, step, reward, term, cp.copy(), bp.copy(),
                  robot(state).copy(), d_live)
        if best is None or d_fixed < best[0]:
            best = record
        if d_fixed < .022 or term or trunc:
            break

    print(mode, "CLOSEST", "step", best[1], "reward", best[2], "term", best[3],
          "fixed_dxy", round(best[0], 6), "live_dxy", round(best[7], 6),
          "cube_pose", np.round(best[4], 6).tolist(),
          "bin_pose", np.round(best[5], 6).tolist(),
          "robot_base_q", np.round(best[6], 6).tolist())

    # At the achieved close state, either simply settle, or raise joint 2 (the
    # calibrated vertical degree) while maintaining a tiny centerward bias.
    hold_base = robot(state)[:3].copy()
    hold_q = robot(state)[3:10].copy()
    for step in range(100):
        goal_q = hold_q.copy()
        if mode != "baseline":
            goal_q[1] = 1.02
            cp = pose(state, name)
            error = target[:2] - cp[:2]
            # Very small bias prevents the lift from merely withdrawing from
            # the rim; cap it well below the normal 12 mm approach steps.
            goal_base = hold_base.copy()
            goal_base[:2] += np.clip(error, -.004, .004)
        else:
            goal_base = hold_base
        action = action_to(state, goal_base, goal_q, 0)
        action[:3] = np.clip(action[:3], -.004, .004)
        action[3:10] = np.clip(action[3:10], -.025, .025)
        state, reward, term, trunc, _ = env.step(action)
        if step % 10 == 0 or reward != -1. or term:
            cp = pose(state, name)
            bp = pose(state, "bin_red")
            print(mode, "AFTER", step, "reward", reward, "term", term,
                  "fixed_dxy", round(float(np.linalg.norm(cp[:2] - target[:2])), 6),
                  "live_dxy", round(float(np.linalg.norm(cp[:2] - bp[:2])), 6),
                  "dz", round(float(cp[2] - bp[2]), 6),
                  "cube", np.round(cp, 5).tolist(), "bin", np.round(bp, 5).tolist())
        if term or trunc:
            break
    env.close()


def run_policy(mode):
    """Reproduce the current controller's known 4.8 cm red placement."""
    env = make_env()
    state, info = env.reset(seed=0, options={"object_count": 4})
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    target = get(state, "bin_red", ["x", "y", "z"]).copy()
    best = None
    reward_changes = []
    previous_reward = None
    for step in range(275):
        state, reward, term, trunc, _ = env.step(policy.get_action(state))
        cp, bp = pose(state, "cube1"), pose(state, "bin_red")
        d_fixed = float(np.linalg.norm(cp[:2] - target[:2]))
        record = (d_fixed, step, reward, term, cp.copy(), bp.copy(),
                  robot(state).copy(), float(np.linalg.norm(cp[:2] - bp[:2])))
        if best is None or d_fixed < best[0]:
            best = record
        if reward != previous_reward:
            reward_changes.append((step, float(reward)))
            previous_reward = reward
        if term or trunc:
            break
    print(mode, "CLOSEST", "step", best[1], "reward", best[2], "term", best[3],
          "fixed_dxy", round(best[0], 6), "live_dxy", round(best[7], 6),
          "cube_pose", np.round(best[4], 6).tolist(),
          "bin_pose", np.round(best[5], 6).tolist(),
          "robot_base_q", np.round(best[6], 6).tolist(),
          "reward_changes", reward_changes)
    cp, bp = pose(state, "cube1"), pose(state, "bin_red")
    print(mode, "END275", "reward", reward, "fixed_dxy",
          round(float(np.linalg.norm(cp[:2] - target[:2])), 6), "live_dxy",
          round(float(np.linalg.norm(cp[:2] - bp[:2])), 6), "cube",
          np.round(cp, 6).tolist(), "bin", np.round(bp, 6).tolist())

    hold_base = robot(state)[:3].copy()
    hold_q = robot(state)[3:10].copy()
    for step in range(100):
        goal_q = hold_q.copy()
        if mode == "policy_lift":
            goal_q[1] = 1.02
            error = target[:2] - pose(state, "cube1")[:2]
            goal_base = hold_base.copy()
            goal_base[:2] += np.clip(error, -.004, .004)
        else:
            goal_base = hold_base
        action = action_to(state, goal_base, goal_q, 0)
        action[:3] = np.clip(action[:3], -.004, .004)
        action[3:10] = np.clip(action[3:10], -.025, .025)
        state, reward, term, trunc, _ = env.step(action)
        if step % 10 == 0 or reward != -1. or term:
            cp, bp = pose(state, "cube1"), pose(state, "bin_red")
            print(mode, "AFTER", step, "reward", reward, "term", term,
                  "fixed_dxy", round(float(np.linalg.norm(cp[:2] - target[:2])), 6),
                  "live_dxy", round(float(np.linalg.norm(cp[:2] - bp[:2])), 6),
                  "dz", round(float(cp[2] - bp[2]), 6),
                  "cube", np.round(cp, 5).tolist(), "bin", np.round(bp, 5).tolist())
        if term or trunc:
            break
    env.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    (run_policy(mode) if mode.startswith("policy_") else run(mode))

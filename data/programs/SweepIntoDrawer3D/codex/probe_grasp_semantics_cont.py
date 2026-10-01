"""Focused wiper grasp probe: closure timing, wrist q7, and base alignment."""

import numpy as np

from env_client import make_env


def drive(env, state, base, joints, grip, steps, drift=(0.0, 0.0)):
    start_wiper = state[147:150].copy()
    for _ in range(steps):
        action = np.zeros(11, np.float32)
        target = np.asarray(base) + np.asarray(drift)
        action[:2] = np.clip((target - state[125:127]) * 0.8, -0.06, 0.06)
        action[3:10] = np.clip((joints - state[128:135]) * 0.6, -0.08, 0.08)
        action[10] = grip
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return state, float(np.linalg.norm(state[147:150] - start_wiper))


def run(name, bx, by, q7, close_while_moving):
    env = make_env()
    try:
        state, _ = env.reset(seed=0)
        initial_wiper = state[147:150].copy()
        joints = state[128:135].copy()
        joints[1] = -0.28
        joints[6] = q7

        # Reach the known local neighborhood with jaws open (0).
        state, _ = drive(env, state, (bx, by - 0.16), joints, 0.0, 90)
        before_approach = state[147:150].copy()
        if close_while_moving:
            state, contact = drive(env, state, (bx, by), joints, 1.0, 12)
        else:
            state, contact = drive(env, state, (bx, by), joints, 0.0, 12)
            state, close_motion = drive(env, state, (bx, by), joints, 1.0, 5)
            contact += close_motion

        before_pull = state[147:150].copy()
        state, follow = drive(env, state, (bx + 0.12, by), joints, 1.0, 12)
        after_pull = state[147:150].copy()
        print(
            name,
            "base", np.round(state[125:127], 3),
            "grip", round(float(state[135]), 3),
            "q7", round(float(state[134]), 3),
            "approach_d", round(float(np.linalg.norm(before_pull - before_approach)), 4),
            "follow", round(follow, 4),
            "total", round(float(np.linalg.norm(after_pull - initial_wiper)), 4),
            "wiper", np.round(after_pull, 3),
        )
    finally:
        env.close()


if __name__ == "__main__":
    cases = [
        ("static_center", 1.30, -0.38, 0.0, False),
        ("moving_center", 1.30, -0.38, 0.0, True),
        ("moving_q7_m04", 1.30, -0.38, -0.4, True),
        ("moving_q7_p04", 1.30, -0.38, 0.4, True),
    ]
    for case in cases:
        run(*case)

"""Test repeated gentle can contacts with fold/reposition/redeploy cycles."""

import argparse

import numpy as np

from env_client import make_env


def servo_pose(env, obs, target, limit=70):
    for _ in range(limit):
        err = target - obs[96:103]
        if np.max(np.abs(err)) < 0.045:
            break
        action = np.zeros(11, dtype=np.float32)
        action[3:10] = np.clip(0.5 * err, -0.1, 0.1)
        obs, *_ = env.step(action)
    return obs


def base_y(env, obs, target, limit=40):
    for _ in range(limit):
        error = target - obs[94]
        if abs(error) < 0.02:
            break
        action = np.zeros(11, dtype=np.float32)
        action[1] = np.clip(0.5 * error, -0.1, 0.1)
        obs, *_ = env.step(action)
    return obs


def run(seed, speed, cycles, q2, q4, yaw_delta, joint1_delta):
    env = make_env()
    obs, _ = env.reset(seed=seed)
    home = obs[96:103].copy()
    push = home.copy()
    push[1], push[3] = q2, q4
    push[0] += joint1_delta
    initial = obs[32:39].copy()
    goal = float(obs[1] - (obs[14] + obs[46] + 0.025))
    print(f"seed={seed} initial={np.round(initial[:3], 4)} goal_y={goal:.4f}")
    for cycle in range(cycles):
        obs = servo_pose(env, obs, home)
        obs = base_y(env, obs, float(obs[33] - 0.28))
        yaw_target = float(obs[95] + yaw_delta)
        for _ in range(20):
            error = yaw_target - obs[95]
            if abs(error) < 0.01:
                break
            action = np.zeros(11, dtype=np.float32)
            action[2] = np.clip(0.5 * error, -0.1, 0.1)
            obs, *_ = env.step(action)
        obs = servo_pose(env, obs, push)
        before = obs[32:39].copy()
        for step in range(70):
            if obs[33] >= goal:
                break
            action = np.zeros(11, dtype=np.float32)
            action[1] = speed
            obs, *_ = env.step(action)
        after = obs[32:39].copy()
        upright = 1.0 - 2.0 * float(after[4] ** 2 + after[5] ** 2)
        print(
            f" cycle={cycle+1} steps={step+1} base_y={obs[94]:.4f} "
            f"dxyz={np.round(after[:3]-before[:3], 4)} "
            f"xyz={np.round(after[:3], 4)} upright={upright:.4f}"
        )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=0.01)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--q2", type=float, default=0.67)
    parser.add_argument("--q4", type=float, default=-1.43)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--joint1", type=float, default=0.0)
    args = parser.parse_args()
    run(args.seed, args.speed, args.cycles, args.q2, args.q4, args.yaw, args.joint1)

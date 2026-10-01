"""Focused black-box probes for stable lateral object contact."""

import argparse

import numpy as np

from env_client import make_env


def drive_pose(env, obs, q_target, base_target, grip, steps):
    for _ in range(steps):
        action = np.zeros(11, dtype=np.float32)
        nbase = len(base_target)
        action[:nbase] = np.clip(np.asarray(base_target) - obs[93 : 93 + nbase], -0.1, 0.1)
        action[3:10] = np.clip(2.5 * (q_target - obs[96:103]), -0.1, 0.1)
        action[10] = grip
        obs, reward, terminated, truncated, info = env.step(action)
    return obs


def run(seed, obj_start, q2, q4, lateral_offset, travel, speed, settle, grip, yaw):
    env = make_env()
    obs, _ = env.reset(seed=seed)
    original = obs[obj_start : obj_start + 7].copy()
    q_target = obs[96:103].copy()
    q_target[1], q_target[3] = q2, q4
    # Rotate the empirically calibrated yaw-zero contact vector with the base.
    local_contact = np.array([obs[obj_start] - obs[93], -lateral_offset])
    c, s = np.cos(yaw), np.sin(yaw)
    world_contact = np.array(
        [c * local_contact[0] - s * local_contact[1],
         s * local_contact[0] + c * local_contact[1]]
    )
    base_xy = obs[obj_start : obj_start + 2] - world_contact
    base_xy[0] = min(base_xy[0], -0.14)  # cabinet collision boundary
    base_target = np.r_[base_xy, yaw]
    obs = drive_pose(env, obs, q_target, base_target, 0.0, 70)
    # Match the known contact timing while closing and continuing pose control.
    obs = drive_pose(env, obs, q_target, base_target, grip, 8)
    # Brake arm motion before making contact; do not keep injecting joint error.
    for _ in range(settle):
        action = np.zeros(11, dtype=np.float32)
        action[10] = grip
        obs, *_ = env.step(action)
    before = obs[obj_start : obj_start + 7].copy()
    target_y = base_target[1] - travel
    for _ in range(20):
        action = np.zeros(11, dtype=np.float32)
        action[1] = np.clip(target_y - obs[94], -speed, speed)
        action[2] = np.clip(yaw - obs[95], -0.1, 0.1)
        if settle == 0:
            action[3:10] = np.clip(2.5 * (q_target - obs[96:103]), -0.1, 0.1)
        action[10] = grip
        obs, *_ = env.step(action)
    after = obs[obj_start : obj_start + 7].copy()
    env.close()
    quat_dot = abs(float(np.dot(before[3:7], after[3:7])))
    print(
        f"seed={seed} q=({q2:.2f},{q4:.2f}) off={lateral_offset:.2f} "
        f"speed={speed:.2f} grip={grip:.0f} base_y={obs[94]:.3f} "
        f"dxyz={np.round(after[:3]-before[:3], 4)} "
        f"from_reset={np.round(after[:3]-original[:3], 4)} qdot={quat_dot:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object", choices=("box", "can"), default="box")
    parser.add_argument("--q2", type=float, default=0.65)
    parser.add_argument("--q4", type=float, default=-1.7)
    parser.add_argument("--offset", type=float, default=0.28)
    parser.add_argument("--travel", type=float, default=0.18)
    parser.add_argument("--speed", type=float, default=0.04)
    parser.add_argument("--settle", type=int, default=30)
    parser.add_argument("--grip", type=float, default=1.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    args = parser.parse_args()
    run(
        args.seed,
        16 if args.object == "box" else 32,
        args.q2,
        args.q4,
        args.offset,
        args.travel,
        args.speed,
        args.settle,
        args.grip,
        args.yaw,
    )

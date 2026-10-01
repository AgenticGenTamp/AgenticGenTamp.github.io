"""Search arm-contact motions that translate the loaded yellow tray toward +y.

This is deliberately separate from the submitted policy.  Each trial starts a
fresh simulator so sign/base-offset comparisons are independent.
"""

import argparse
import math

import numpy as np

from env_client import make_env


def xyz(state, name):
    obj = state.get_object_from_name(name)
    return np.array([state.get(obj, f) for f in ("x", "y", "z")], float)


def base_pose(state):
    obj = state.get_object_from_name("robot")
    return np.array([state.get(obj, f) for f in
                     ("pos_base_x", "pos_base_y", "pos_base_rot")], float)


def cubes(state):
    return np.array([xyz(state, n) for n in state.get_object_names()
                     if n.startswith("cube_")])


def containment(state):
    """Count cube centers conservatively inside the yellow bin footprint."""
    center = xyz(state, "bin_yellow_0")
    points = cubes(state)
    # Observed bb is full size 0.45 x 0.30. Leave 1cm for the inner wall.
    inside = ((np.abs(points[:, 0] - center[0]) < 0.215) &
              (np.abs(points[:, 1] - center[1]) < 0.140) &
              (points[:, 2] > center[2] - 0.01))
    return int(inside.sum()), len(points), points.mean(axis=0)


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def legacy_prep(env, state, steps):
    """Recreate af78e0e's 79-step contact setup without changing approach.py."""
    robot = state.get_object_from_name("robot")
    scoop = state.get_object_from_name("scoop_0")
    source = state.get_object_from_name("bin_yellow_0")
    home = np.array([state.get(robot, "pos_arm_joint%d" % i)
                     for i in range(1, 8)])
    sp = xyz(state, "scoop_0")[:2]
    align = np.array([sp[0] - 0.417, sp[1] + 0.003,
                      math.atan2(-0.003, 0.417)])
    src = xyz(state, "bin_yellow_0")[:2]
    source_dir = src - sp
    source_dir /= max(np.linalg.norm(source_dir), 1e-6)
    rewards = []
    for t in range(1, steps + 1):
        action = np.zeros(11, np.float32)
        if t <= 30:
            error = align - base_pose(state)
            error[2] = wrap(error[2])
            action[:3] = np.clip(0.55 * error, -0.1, 0.1)
            joints = np.array([state.get(robot, "pos_arm_joint%d" % i)
                               for i in range(1, 8)])
            action[3:10] = np.clip(0.7 * (home - joints), -0.1, 0.1)
        elif t <= 62:
            action[4], action[6] = 0.1, 0.06
        elif t <= 67:
            action[8] = 0.1
        else:
            action[:2] = np.clip(0.035 * source_dir, -0.05, 0.05)
            action[2] = -0.1 if source_dir[1] < 0.0 else 0.1
        action[10] = 1.0
        state, reward, term, trunc, _ = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break
    return state, rewards


def run(seed, count, dx, dy, joint3, joint4, press_steps, settle_steps=8,
        prep_steps=79, cycles=1, follow=False, dyaw=0.0,
        second_joint3=0, second_joint4=0):
    env = make_env()
    options = None if count < 0 else {"object_count": count}
    state, _ = env.reset(seed=seed, options=options)
    yellow_initial = xyz(state, "bin_yellow_0")
    green = xyz(state, "bin_green_0")
    rewards = []

    # Reproduce the calibrated scoop/source approach which was found to put
    # the wrist against the near wall of the loaded yellow tray.
    state, prep_rewards = legacy_prep(env, state, prep_steps)
    rewards.extend(prep_rewards)
    yellow0 = xyz(state, "bin_yellow_0")
    cube0 = cubes(state)
    pose0 = base_pose(state)

    # Shift the entire kinematic chain while maintaining its existing contact.
    target = pose0.copy()
    target += (dx, dy, dyaw)
    for _ in range(50):
        error = target - base_pose(state)
        error[2] = wrap(error[2])
        if np.max(np.abs(error)) < 0.008:
            break
        action = np.zeros(11, np.float32)
        action[:3] = np.clip(1.5 * error, -0.1, 0.1)
        action[10] = 1.0
        state, reward, term, trunc, _ = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break

    yellow_after_offset = xyz(state, "bin_yellow_0")
    reached_offset = base_pose(state) - pose0
    # Arm joint 3/4 correspond to action indices 5/6 after the three base DOFs.
    for cycle in range(cycles):
        cycle_bin0 = xyz(state, "bin_yellow_0")
        for _ in range(press_steps):
            action = np.zeros(11, np.float32)
            action[5] = 0.1 * joint3
            action[6] = 0.1 * joint4
            action[10] = 1.0
            state, reward, term, trunc, _ = env.step(action)
            rewards.append(reward)
            if term or trunc:
                break
        # Retract along the same joint path: the gripper is on the non-pushing
        # side, so this should reset the arm without pulling the tray back.
        if cycle + 1 < cycles:
            for _ in range(press_steps):
                action = np.zeros(11, np.float32)
                action[5] = -0.1 * joint3
                action[6] = -0.1 * joint4
                action[10] = 1.0
                state, reward, term, trunc, _ = env.step(action)
                rewards.append(reward)
                if term or trunc:
                    break
            if follow:
                # Restore the original arm/tray contact geometry in world
                # space by following the tray's measured displacement.
                shift = xyz(state, "bin_yellow_0")[:2] - cycle_bin0[:2]
                follow_target = base_pose(state)[:2] + shift
                for _ in range(35):
                    error = follow_target - base_pose(state)[:2]
                    if np.max(np.abs(error)) < 0.006:
                        break
                    action = np.zeros(11, np.float32)
                    action[:2] = np.clip(1.5 * error, -0.1, 0.1)
                    action[10] = 1.0
                    state, reward, term, trunc, _ = env.step(action)
                    rewards.append(reward)
                    if term or trunc:
                        break
    if second_joint3 or second_joint4:
        for _ in range(press_steps):
            action = np.zeros(11, np.float32)
            action[5] = 0.1 * second_joint3
            action[6] = 0.1 * second_joint4
            action[10] = 1.0
            state, reward, term, trunc, _ = env.step(action)
            rewards.append(reward)
            if term or trunc:
                break
    for _ in range(settle_steps):
        action = np.zeros(11, np.float32)
        action[10] = 1.0
        state, reward, term, trunc, _ = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break

    yellow1 = xyz(state, "bin_yellow_0")
    cube1 = cubes(state)
    held, total, cube_mean = containment(state)
    result = {
        "seed": seed,
        "dx": dx, "dy": dy, "dyaw": dyaw, "j3": joint3, "j4": joint4,
        "cycles": cycles,
        "second_joint3": second_joint3, "second_joint4": second_joint4,
        "follow": follow, "reached_offset": reached_offset,
        "bin_delta": yellow1 - yellow0,
        "prep_delta": yellow0 - yellow_initial,
        "offset_delta": yellow_after_offset - yellow0,
        "press_delta": yellow1 - yellow_after_offset,
        "goal_dist": float(np.linalg.norm(yellow1[:2] - green[:2])),
        "held": held, "count": total,
        "cube_mean_delta": cube1.mean(axis=0) - cube0.mean(axis=0),
        "cube_mean": cube_mean, "reward_sum": float(sum(rewards)),
        "reward_max": float(max(rewards, default=0.0)), "steps": len(rewards),
    }
    env.close()
    return result


def show(r):
    print("seed=%d offset=(%+.2f,%+.2f,%+.2f)->%s "
          "joints=(%+d,%+d)x%d then=(%+d,%+d) follow=%d steps=%d "
          "dbin=%s [base=%s press=%s] dcubes=%s held=%d/%d "
          "goal=%.3f R=%.2f maxR=%.2f" %
          (r["seed"], r["dx"], r["dy"], r["dyaw"],
           np.array2string(r["reached_offset"], precision=3),
           r["j3"], r["j4"], r["cycles"], r["second_joint3"],
           r["second_joint4"], r["follow"], r["steps"],
           np.array2string(r["bin_delta"], precision=3),
           np.array2string(r["offset_delta"], precision=3),
           np.array2string(r["press_delta"], precision=3),
           np.array2string(r["cube_mean_delta"], precision=3),
           r["held"], r["count"], r["goal_dist"], r["reward_sum"],
           r["reward_max"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dyaw", type=float, default=0.0)
    parser.add_argument("--j3", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--j4", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--second-j3", type=int, choices=(-1, 0, 1), default=0)
    parser.add_argument("--second-j4", type=int, choices=(-1, 0, 1), default=0)
    args = parser.parse_args()
    if args.grid:
        # Lateral shifts straddle both long sides of the tray; two small x
        # shifts test whether contact is made at its near or far edge.
        configs = [(dx, dy, j3, j4)
                   for dx in (-0.08, 0.0, 0.08)
                   for dy in (-0.12, 0.0, 0.12)
                   for j3, j4 in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    else:
        configs = [(args.dx, args.dy, args.j3, args.j4)]
    results = []
    for config in configs:
        result = run(args.seed, args.count, *config, args.steps,
                     cycles=args.cycles, follow=args.follow, dyaw=args.dyaw,
                     second_joint3=args.second_j3,
                     second_joint4=args.second_j4)
        show(result)
        results.append(result)
    print("BEST +y with all cubes")
    valid = [r for r in results if r["held"] == r["count"]]
    for result in sorted(valid, key=lambda r: r["bin_delta"][1], reverse=True)[:5]:
        show(result)


if __name__ == "__main__":
    main()

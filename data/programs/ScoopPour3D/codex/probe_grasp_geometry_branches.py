"""Classify reset geometry and probe the corresponding grasp branch.

This is an experiment only; it deliberately does not import approach.py.  The
reported residual is useful for deciding whether a reset belongs to the
original arm-line calibration or needs the canonical world-x reach pose.
"""

import argparse
import math

import numpy as np

from env_client import make_env


def values(state, name, features):
    obj = state.get_object_from_name(name)
    return np.asarray([state.get(obj, f) for f in features], dtype=float)


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def run(seed, force_branch=None):
    env = make_env()
    state, _ = env.reset(seed=seed)
    base0 = values(state, "robot", ("pos_base_x", "pos_base_y", "pos_base_rot"))
    joints0 = values(state, "robot", tuple("pos_arm_joint%d" % i for i in range(1, 8)))
    scoop0 = values(state, "scoop_0", ("x", "y", "z", "qw", "qx", "qy", "qz"))
    q = scoop0[3:]
    yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                     1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))
    axis = np.array([math.cos(yaw), math.sin(yaw)])
    reach_residual = np.linalg.norm(scoop0[:2] - base0[:2] - [0.417, 0.0])
    branch = force_branch or ("arm_line" if reach_residual < 0.035 else "canonical")

    # Empirically found calibration.  arm_line is the seed-0 family.  The
    # canonical parameters retain the scoop on seed 2 despite its rotated and
    # laterally displaced reset.  Seed 1/4 are intentionally useful negative
    # controls: contact/lift without lateral retention is not called a grasp.
    offset = 0.060
    if branch == "arm_line":
        target_xy = base0[:2] + offset * axis
        target_yaw = base0[2]
        wrist_delta = 0.0
    else:
        target_xy = scoop0[:2] - np.array([0.417, 0.0]) + offset * axis
        target_yaw = math.atan2(scoop0[1] - target_xy[1],
                                scoop0[0] - target_xy[0])
        wrist_delta = wrap(yaw - target_yaw + 0.557)
        if wrist_delta > math.pi / 2:
            wrist_delta -= math.pi
        elif wrist_delta < -math.pi / 2:
            wrist_delta += math.pi
    target_wrist = joints0[6] + wrist_delta

    for _ in range(12):
        base = values(state, "robot", ("pos_base_x", "pos_base_y", "pos_base_rot"))
        wrist = values(state, "robot", ("pos_arm_joint7",))[0]
        action = np.zeros(11, np.float32)
        action[:2] = np.clip(0.8 * (target_xy - base[:2]), -0.1, 0.1)
        action[2] = np.clip(0.8 * wrap(target_yaw - base[2]), -0.1, 0.1)
        action[9] = np.clip(0.8 * (target_wrist - wrist), -0.1, 0.1)
        action[10] = 1.0
        state, _, _, _, _ = env.step(action)
    for i in range(32):
        action = np.zeros(11, np.float32)
        action[4], action[6] = 0.1, 0.06
        action[10] = 0.0 if i >= 30 else 1.0
        state, _, _, _, _ = env.step(action)
    action = np.zeros(11, np.float32)
    action[10] = 0.0
    for _ in range(8):
        state, _, _, _, _ = env.step(action)
    action[4], action[6] = -0.1, -0.06
    for _ in range(18):
        state, _, _, _, _ = env.step(action)
    after_lift = values(state, "scoop_0", ("x", "y", "z"))
    action[:] = 0.0
    action[1], action[10] = 0.06, 0.0
    for _ in range(4):
        state, _, _, _, _ = env.step(action)
    final = values(state, "scoop_0", ("x", "y", "z"))
    base_final = values(state, "robot", ("pos_base_x", "pos_base_y"))
    lift = after_lift[2] - scoop0[2]
    scoop_follow = np.linalg.norm(final[:2] - after_lift[:2])
    base_follow = np.linalg.norm(base_final - target_xy)
    retained = lift > 0.04 and scoop_follow > 0.5 * base_follow
    print("seed=%d branch=%s residual=%.4f rel=%s yaw=%.4f" %
          (seed, branch, reach_residual,
           np.array2string(scoop0[:2] - base0[:2], precision=4), yaw))
    print("target_xy=%s target_yaw=%.4f wrist_delta=%.4f offset=%.3f" %
          (np.array2string(target_xy, precision=4), target_yaw, wrist_delta, offset))
    print("lift=%.4f scoop_follow=%.4f base_follow=%.4f retained=%s" %
          (lift, scoop_follow, base_follow, retained))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--branch", choices=("arm_line", "canonical"))
    args = parser.parse_args()
    run(args.seed, args.branch)

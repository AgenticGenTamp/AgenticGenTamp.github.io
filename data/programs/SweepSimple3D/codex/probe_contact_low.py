"""Random arm-only wiper contact search from chassis-safe ring poses."""
import math
import sys

import numpy as np

from env_client import make_env


def val(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


def xyz(state, name):
    return np.array([val(state, name, f) for f in "xyz"])


def base(state):
    return np.array([val(state, "robot", "pos_base_x"),
                     val(state, "robot", "pos_base_y")])


def joints(state):
    return np.array([val(state, "robot", f"pos_arm_joint{i}")
                     for i in range(1, 8)])


def angle_error(target, actual):
    return (target - actual + math.pi) % (2 * math.pi) - math.pi


def move_base(env, state, goal, yaw, qhome):
    for _ in range(45):
        action = np.zeros(11, np.float32)
        action[:2] = np.clip(goal - base(state), -.1, .1)
        action[2] = np.clip(angle_error(yaw, val(state, "robot", "pos_base_rot")), -.1, .1)
        action[3:10] = np.clip(.7 * (qhome - joints(state)), -.1, .1)
        action[10] = 1.
        state, _, _, _, _ = env.step(action)
        if np.max(np.abs(goal-base(state))) < .012:
            break
    # Explicit all-zero-base settling period.
    for _ in range(4):
        action = np.zeros(11, np.float32); action[10] = 1.
        state, _, _, _, _ = env.step(action)
    return state


worker = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(7717 + worker)
env = make_env()
grip = 0. if worker >= 2 else 1.
bearings = ([.30*math.pi, .50*math.pi, .70*math.pi] if worker == 0
            else [.40*math.pi, .60*math.pi, .80*math.pi])

for pose_i, bearing in enumerate(bearings):
    state, _ = env.reset(seed=0, options={"object_count": 1})
    wiper = xyz(state, "wiper_0")
    qhome = joints(state)
    radius = [.355, .42, .52][pose_i]
    goal = wiper[:2] + radius*np.array([math.cos(bearing), math.sin(bearing)])
    yaw = math.atan2(wiper[1]-goal[1], wiper[0]-goal[0])
    state = move_base(env, state, goal, yaw, qhome)
    print("POSE", worker, pose_i, "base", base(state).round(4).tolist(),
          "yaw", round(val(state, "robot", "pos_base_rot"), 4),
          "wiper", xyz(state, "wiper_0").round(4).tolist(),
          "dist", round(float(np.linalg.norm(base(state)-xyz(state, "wiper_0")[:2])), 4), flush=True)

    for trial in range(16):
        # Bias shoulder and elbow toward the empirically low-reaching family.
        target = np.array([rng.uniform(-1.2, 1.2),
                           rng.uniform(-1.65, .30),
                           rng.normal(2.37, .55),
                           rng.uniform(-3.0, -1.55),
                           rng.uniform(-1.2, 1.2),
                           rng.uniform(-1.55, -.2),
                           rng.uniform(-math.pi, math.pi)])
        before = xyz(state, "wiper_0")
        qbefore = joints(state).copy()
        max_step = 0.
        prev = before.copy()
        min_dist = 99.
        for step in range(18):
            action = np.zeros(11, np.float32)  # chassis command stays exactly zero
            action[3:10] = np.clip(.72*(target-joints(state)), -.1, .1)
            action[10] = grip
            state, _, _, _, _ = env.step(action)
            now = xyz(state, "wiper_0")
            max_step = max(max_step, float(np.linalg.norm(now-prev)))
            min_dist = min(min_dist, float(np.linalg.norm(base(state)-now[:2])))
            prev = now
        delta = xyz(state, "wiper_0")-before
        amount = float(np.linalg.norm(delta))
        if amount > .002 or max_step > .002:
            print("CONTACT", "worker", worker, "pose", pose_i, "trial", trial,
                  "base", base(state).round(5).tolist(),
                  "yaw", round(val(state, "robot", "pos_base_rot"), 5),
                  "q_before", qbefore.round(5).tolist(),
                  "q_target", target.round(5).tolist(),
                  "q_actual", joints(state).round(5).tolist(),
                  "w_before", before.round(5).tolist(),
                  "w_after", xyz(state, "wiper_0").round(5).tolist(),
                  "delta", delta.round(5).tolist(),
                  "amount", round(amount, 5), "max_step", round(max_step, 5),
                  "min_base_dist", round(min_dist, 5), flush=True)
env.close()

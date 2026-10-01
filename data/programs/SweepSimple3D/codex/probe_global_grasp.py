"""Global randomized wiper-grasp search with arm-only tug validation."""
import math
import numpy as np
from env_client import make_env


def value(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


def xyz(state, name):
    return np.array([value(state, name, f) for f in "xyz"])


def base(state):
    return np.array([value(state, "robot", "pos_base_x"),
                     value(state, "robot", "pos_base_y")])


def joints(state):
    return np.array([value(state, "robot", f"pos_arm_joint{i}")
                     for i in range(1, 8)])


def angle_error(target, actual):
    return (target - actual + math.pi) % (2 * math.pi) - math.pi


def command_to(env, state, bp, yaw, qtarget, grip, limit=34):
    for _ in range(limit):
        action = np.zeros(11, np.float32)
        action[:2] = np.clip(bp - base(state), -.1, .1)
        action[2] = np.clip(angle_error(yaw, value(state, "robot", "pos_base_rot")), -.1, .1)
        action[3:10] = np.clip(.65 * (qtarget - joints(state)), -.1, .1)
        action[10] = grip
        state, _, _, _, _ = env.step(action)
        if (np.max(np.abs(qtarget - joints(state))) < .035 and
                np.max(np.abs(bp - base(state))) < .02):
            break
    return state


def tug(env, state, qtarget, dim, delta):
    before = xyz(state, "wiper_0")
    goal = qtarget.copy(); goal[dim] += delta
    for _ in range(7):
        action = np.zeros(11, np.float32)
        action[3:10] = np.clip(.8 * (goal - joints(state)), -.1, .1)
        action[10] = 0.
        state, _, _, _, _ = env.step(action)
    return state, xyz(state, "wiper_0") - before


if __name__ == "__main__":
    rng = np.random.default_rng(90210)
    env = make_env()
    # Broad mechanically plausible ranges. q3 is centered around its home pi
    # because this joint mainly rolls the arm; q4/q6 provide reach and height.
    lo = np.array([-3.0, -1.65, 1.35, -2.95, -2.2, -1.8, -3.0])
    hi = np.array([ 3.0,  1.35, 4.90,  0.25,  2.2,  1.2,  3.0])
    for trial in range(32):
        state, _ = env.reset(seed=0, options={"object_count": 1})
        w0 = xyz(state, "wiper_0")
        bearing = rng.uniform(-math.pi, math.pi)
        radius = rng.uniform(.30, .78)
        bp = w0[:2] + radius * np.array([math.cos(bearing), math.sin(bearing)])
        yaw = angle_error(bearing + math.pi + rng.normal(0, .28), 0.)
        qt = rng.uniform(lo, hi)
        # Gripper control is an absolute target: 1=open and 0=closed.
        state = command_to(env, state, bp, yaw, qt, 1.)
        # Two no-motion settling steps make incidental collision distinguishable
        # from motion caused after closure.
        for _ in range(2):
            action = np.zeros(11, np.float32); action[10] = 1.
            state, _, _, _, _ = env.step(action)
        at_close = xyz(state, "wiper_0")
        for _ in range(7):
            action = np.zeros(11, np.float32); action[10] = 0.
            state, _, _, _, _ = env.step(action)
        closed = xyz(state, "wiper_0")
        # Shoulder and elbow tugs move the arm substantially with zero chassis
        # command. A held handle should follow both.
        state, d2 = tug(env, state, joints(state), 1, .45)
        state, d4 = tug(env, state, joints(state), 3, .40)
        final = xyz(state, "wiper_0")
        score = max(np.linalg.norm(d2), np.linalg.norm(d4), final[2] - w0[2])
        print(trial, "r/b/y", np.round([radius, bearing, yaw], 3).tolist(),
              "q", np.round(qt, 3).tolist(), "actual", np.round(joints(state), 3).tolist(),
              "place", np.round(at_close-w0, 4).tolist(),
              "close", np.round(closed-at_close, 4).tolist(),
              "tug2", np.round(d2, 4).tolist(), "tug4", np.round(d4, 4).tolist(),
              "final", np.round(final-w0, 4).tolist(), "score", round(float(score), 4),
              flush=True)
        if final[2] - w0[2] > .04 or (np.linalg.norm(d2) > .04 and np.linalg.norm(d4) > .04):
            print("STRONG CANDIDATE", trial, flush=True)
            break
    env.close()

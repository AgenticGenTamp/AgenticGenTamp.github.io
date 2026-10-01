"""Probe action[10] gripper semantics and slow contact trajectories."""

import math

import numpy as np

from env_client import make_env


HOME = np.asarray([0.0, -0.3491, math.pi, -2.5482, 0.0, -0.8727,
                   math.pi / 2])


def read(state, name, features):
    obj = state.get_object_from_name(name)
    return np.asarray([float(state.get(obj, feature)) for feature in features])


def robot(state):
    q = read(state, "robot", tuple("pos_arm_joint%d" % i for i in range(1, 8)))
    grip = read(state, "robot", ("pos_gripper", "vel_gripper"))
    base = read(state, "robot", ("pos_base_x", "pos_base_y", "pos_base_rot"))
    return base, q, grip


def cube(state):
    return read(state, "cube_0", ("x", "y", "z"))


def step(env, action):
    state, reward, terminated, truncated, _ = env.step(action)
    return state, reward, terminated or truncated


def temporal():
    env = make_env()
    try:
        for command in (0.0, 1.0, 0.25, 0.75):
            state, _ = env.reset(seed=0, options={"object_count": 1})
            samples = []
            for i in range(20):
                action = np.zeros(18, dtype=np.float32)
                action[10] = command
                state, _, _ = step(env, action)
                if i in (0, 1, 2, 4, 9, 19):
                    samples.append((i + 1, *robot(state)[2]))
            print("TEMP", command, np.round(samples, 5).tolist(), flush=True)

        state, _ = env.reset(seed=0, options={"object_count": 1})
        for command in (1.0, 0.0, 1.0):
            samples = []
            for i in range(12):
                action = np.zeros(18, dtype=np.float32)
                action[10] = command
                state, _, _ = step(env, action)
                if i in (0, 1, 3, 7, 11):
                    samples.append((i + 1, *robot(state)[2]))
            print("SWITCH", command, np.round(samples, 5).tolist(), flush=True)
    finally:
        env.close()


def velocity_interaction():
    """Compare position increments with zero, matching, and NaN velocities."""
    for label, velocity in (("zero", 0.0), ("matching", 1.0), ("nan", np.nan)):
        env = make_env()
        try:
            state, _ = env.reset(seed=0, options={"object_count": 1})
            q0 = robot(state)[1]
            samples = []
            for i in range(3):
                action = np.zeros(18, dtype=np.float32)
                action[3] = 0.05
                action[10] = 0.0
                action[11] = velocity
                try:
                    state, _, _ = step(env, action)
                except Exception as error:
                    print("VELOCITY", label, "ERROR", type(error).__name__, str(error),
                          flush=True)
                    break
                samples.append(robot(state)[1][0] - q0[0])
            else:
                print("VELOCITY", label, np.round(samples, 6).tolist(), flush=True)
        finally:
            env.close()


def run_slow_contact(target_q, offsets, seed=0):
    """Try an IK candidate with gradual descent and gradual gripper closure."""
    env = make_env()
    try:
        for xoff, yoff in offsets:
            state, _ = env.reset(seed=seed, options={"object_count": 1})
            start = cube(state)
            base_goal = start[:2] - np.asarray((xoff, yoff))

            # First move the base and arm near the candidate with the claw open.
            for _ in range(55):
                base_now, q_now, _ = robot(state)
                action = np.zeros(18, dtype=np.float32)
                action[:2] = np.clip(base_goal - base_now[:2], -0.05, 0.05)
                # Stop slightly above the IK target, then descend slowly below.
                above = np.asarray(target_q).copy()
                above[1] -= 0.12
                error = (above - q_now + np.pi) % (2 * np.pi) - np.pi
                action[3:10] = np.clip(error, -0.035, 0.035)
                action[10] = 0.0
                # Velocity assistance is useful for the bulk move; it is
                # deliberately removed for the contact-sensitive descent.
                action[11:18] = np.clip(4.0 * error, -2.0, 2.0)
                state, _, _ = step(env, action)

            before = cube(state)
            # Shoulder-2 provides the principal slow descent for this IK family.
            for i in range(24):
                _, q_now, _ = robot(state)
                below = np.asarray(target_q).copy()
                below[1] += 0.08
                error = (below - q_now + np.pi) % (2 * np.pi) - np.pi
                action = np.zeros(18, dtype=np.float32)
                action[3:10] = np.clip(error, -0.012, 0.012)
                action[10] = min(1.0, max(0.0, (i - 5) / 12.0))
                state, _, _ = step(env, action)
                shift = float(np.linalg.norm(cube(state) - before))
                if shift > 1e-4:
                    print("SLOW_CONTACT", (xoff, yoff), "step", i + 1,
                          "shift", shift, "cube", np.round(cube(state), 5).tolist(),
                          "q", np.round(robot(state)[1], 4).tolist(), flush=True)
                    return True

            # A pickup test distinguishes contact/grasp from an inert pose.
            closed = cube(state)
            grasp_q = robot(state)[1].copy()
            for _ in range(25):
                _, q_now, _ = robot(state)
                error = (HOME - q_now + np.pi) % (2 * np.pi) - np.pi
                action = np.zeros(18, dtype=np.float32)
                action[3:10] = np.clip(error, -0.04, 0.04)
                action[10] = 1.0
                state, _, _ = step(env, action)
                shift = float(np.linalg.norm(cube(state) - closed))
                if shift > 1e-4:
                    print("SLOW_PICKUP", (xoff, yoff), "shift", shift,
                          "cube", np.round(cube(state), 5).tolist(), flush=True)
                    return True
            print("SLOW_NONE", (xoff, yoff),
                  "grasp_q", np.round(grasp_q, 3).tolist(),
                  "q", np.round(robot(state)[1], 3).tolist(),
                  "grip", np.round(robot(state)[2], 3).tolist(), flush=True)
    finally:
        env.close()
    return False


def main():
    temporal()
    velocity_interaction()


if __name__ == "__main__":
    main()

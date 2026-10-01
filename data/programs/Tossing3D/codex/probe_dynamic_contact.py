"""Search for dynamic arm contact/launch using saturated joint velocities."""

import numpy as np

from env_client import make_env


Q = np.asarray([0.23, 0.39, 3.05, 0.0, 0.17, -1.57, 2.46])
OFFSET = np.asarray([0.34, 0.12])
EPS = 1e-5


def read(state, name, features):
    obj = state.get_object_from_name(name)
    return np.asarray([float(state.get(obj, f)) for f in features])


def cube(state):
    return read(state, "cube_0", ("x", "y", "z"))


def base(state):
    return read(state, "robot", ("pos_base_x", "pos_base_y"))


def qpos(state):
    return read(state, "robot", tuple("pos_arm_joint%d" % i for i in range(1, 8)))


def advance(env, state, action):
    state, _, terminated, truncated, _ = env.step(action)
    return state, terminated or truncated


def align(env, state, grip, steps):
    for _ in range(steps):
        action = np.zeros(18, dtype=np.float32)
        action[3:10] = np.clip(Q - qpos(state), -0.1, 0.1)
        action[10] = grip
        state, _ = advance(env, state, action)
    return state


def initialize(env, grip):
    state, _ = env.reset(seed=0)
    goal = cube(state)[:2] - OFFSET
    remaining = goal - base(state)
    while np.max(np.abs(remaining)) > 1e-7:
        inc = np.clip(remaining, -0.05, 0.05)
        action = np.zeros(18, dtype=np.float32)
        action[:2] = inc
        action[10] = grip
        state, _ = advance(env, state, action)
        remaining -= inc
    for _ in range(15):
        action = np.zeros(18, dtype=np.float32)
        action[10] = grip
        state, _ = advance(env, state, action)
    return align(env, state, grip, 70)


def patterns():
    result = []
    for joint in range(7):
        for sign in (-1.0, 1.0):
            v = np.zeros(7)
            v[joint] = 10.0 * sign
            result.append(("j%d_%+d" % (joint + 1, int(10 * sign)), v))
    fixed = (
        ("all_pos", [1, 1, 1, 1, 1, 1, 1]),
        ("all_neg", [-1, -1, -1, -1, -1, -1, -1]),
        ("alternating_a", [1, -1, 1, -1, 1, -1, 1]),
        ("alternating_b", [-1, 1, -1, 1, -1, 1, -1]),
        ("distal_pos", [0, 0, 0, 1, 1, 1, 1]),
        ("distal_neg", [0, 0, 0, -1, -1, -1, -1]),
        ("swing_a", [1, 1, -1, -1, 1, 1, -1]),
        ("swing_b", [-1, -1, 1, 1, -1, -1, 1]),
    )
    result.extend((name, 10.0 * np.asarray(signs, dtype=float))
                  for name, signs in fixed)
    rng = np.random.RandomState(20260915)
    for i in range(8):
        result.append(("random_%02d" % i,
                       10.0 * rng.choice((-1.0, 1.0), size=7)))
    return result


def main():
    env = make_env()
    for grip in (0.0, 1.0):
        state = initialize(env, grip)
        reference = cube(state)
        for index, (name, velocity) in enumerate(patterns()):
            q_before = qpos(state).copy()
            for burst_step in range(3):
                action = np.zeros(18, dtype=np.float32)
                action[10] = grip
                action[11:18] = velocity
                state, _ = advance(env, state, action)
                shift = float(np.linalg.norm(cube(state) - reference))
                if shift > EPS:
                    print("FOUND", "grip", grip, "pattern", name,
                          "velocity", velocity.tolist(), "burst_step", burst_step + 1,
                          "shift", shift, "q_before", np.round(q_before, 4),
                          "q_after", np.round(qpos(state), 4),
                          "cube", np.round(cube(state), 6))
                    env.close()
                    return
            # Recover the same local configuration before the next burst.
            state = align(env, state, grip, 45)
            shift = float(np.linalg.norm(cube(state) - reference))
            if shift > EPS:
                print("FOUND during_recovery", "grip", grip, "after", name,
                      "velocity", velocity.tolist(), "shift", shift,
                      "cube", np.round(cube(state), 6))
                env.close()
                return
        print("NO_MOVE grip", grip, "patterns", len(patterns()))
    env.close()


if __name__ == "__main__":
    main()

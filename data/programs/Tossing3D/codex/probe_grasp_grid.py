"""Sweep cube-relative base offsets for a candidate collision-limited grasp."""

import math

import numpy as np

from env_client import make_env


Q_COMMAND = np.asarray([0.0, 0.65, math.pi, -0.24, 0.0, -2.25, math.pi / 2])
# Physical pose reported when that command is collision-limited.  Servo to the
# observed pose directly so every grid point tests the same gripper geometry.
Q_GOAL = np.asarray([0.0, 0.058, math.pi, -0.961, 0.0, -1.656, math.pi / 2])
X_OFFSETS = (0.42, 0.46, 0.50, 0.54, 0.58)
Y_OFFSETS = (-0.08, -0.04, 0.0, 0.04, 0.08)


def read(state, name, features):
    obj = state.get_object_from_name(name)
    return np.asarray([float(state.get(obj, f)) for f in features])


def robot_base(state):
    return read(state, "robot", ("pos_base_x", "pos_base_y"))


def robot_q(state):
    return read(state, "robot", tuple("pos_arm_joint%d" % i for i in range(1, 8)))


def cube_xyz(state):
    return read(state, "cube_0", ("x", "y", "z"))


def step(env, action):
    state, reward, terminated, truncated, info = env.step(action)
    return state, reward, terminated or truncated


def increment_target(env, state, indices, delta, grip, max_increment=0.1):
    """Apply action increments whose sum equals delta, then return state."""
    remaining = np.asarray(delta, dtype=float).copy()
    while np.max(np.abs(remaining)) > 1e-7:
        inc = np.clip(remaining, -max_increment, max_increment)
        action = np.zeros(18, dtype=np.float32)
        action[10] = grip
        action[np.asarray(indices)] = inc
        state, _, done = step(env, action)
        remaining -= inc
        if done:
            break
    return state


def zero_steps(env, state, count, grip):
    action = np.zeros(18, dtype=np.float32)
    action[10] = grip
    rewards = 0.0
    for _ in range(count):
        state, reward, done = step(env, action)
        rewards += reward
        if done:
            break
    return state, rewards


def drive_arm_to_collision(env, state, grip, count=90):
    """Continuously servo toward Q_GOAL; collision prevents full convergence."""
    for _ in range(count):
        error = Q_GOAL - robot_q(state)
        action = np.zeros(18, dtype=np.float32)
        action[3:10] = np.clip(error, -0.1, 0.1)
        action[10] = grip
        state, _, done = step(env, action)
        if done:
            break
    return state


def run(env, xoff, yoff):
    state, _ = env.reset(seed=0)
    cube0 = cube_xyz(state)
    base0 = robot_base(state)
    q0 = robot_q(state)

    # Here offsets mean cube position minus base position.  Thus positive X is
    # a cube in front of the base, matching the default geometry.
    base_goal = cube0[:2] - np.asarray([xoff, yoff])
    state = increment_target(env, state, (0, 1), base_goal - base0, grip=0.0,
                             max_increment=0.05)
    state, _ = zero_steps(env, state, 20, grip=0.0)

    # Unlike the base increments, arm commands must be applied continuously;
    # repeatedly servo toward q goal until contact limits further progress.
    state = drive_arm_to_collision(env, state, grip=0.0)
    actual_q = robot_q(state)
    cube_preclose = cube_xyz(state)

    # Close fully, first checking whether closing itself contacts the cube.
    state, _ = zero_steps(env, state, 15, grip=1.0)
    cube_closed = cube_xyz(state)
    close_shift = np.linalg.norm(cube_closed - cube_preclose)

    # Retreat the base in world -X. If grasped, the cube should follow promptly.
    first_move_shift = None
    remaining = -0.25
    for i in range(5):
        action = np.zeros(18, dtype=np.float32)
        action[0] = -0.05
        action[10] = 1.0
        state, _, _ = step(env, action)
        remaining += 0.05
        if i == 0:
            first_move_shift = np.linalg.norm(cube_xyz(state) - cube_closed)
    state, reward_tail = zero_steps(env, state, 15, grip=1.0)
    cube_final = cube_xyz(state)
    move_shift = np.linalg.norm(cube_final - cube_closed)
    return (actual_q, robot_base(state), cube_preclose, close_shift,
            first_move_shift, move_shift, cube_final, reward_tail)


def main():
    env = make_env()
    best = []
    for xoff in X_OFFSETS:
        for yoff in Y_OFFSETS:
            result = run(env, xoff, yoff)
            q, base, cube_before, close_d, first_d, move_d, cubef, reward = result
            print("offset=(%.2f,%+.2f)" % (xoff, yoff),
                  "q=" + np.array2string(q, precision=2),
                  "close=%.4f" % close_d, "first=%.4f" % first_d,
                  "after_move=%.4f" % move_d,
                  "cube=" + np.array2string(cubef, precision=3),
                  "base=" + np.array2string(base, precision=3),
                  "tail_reward=%.1f" % reward)
            best.append((move_d, close_d, xoff, yoff))
    env.close()
    print("largest movement", sorted(best, reverse=True)[:5])


if __name__ == "__main__":
    main()

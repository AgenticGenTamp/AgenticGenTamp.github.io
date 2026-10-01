"""Broad cube-relative base sweep for grasping at the HOME arm pose."""

import math
import numpy as np

from env_client import make_env


HOME = np.array([0.0, -0.3491, math.pi, -2.5482, 0.0, -0.8727, math.pi / 2])


def value(state, obj, feature):
    return float(state.get(obj, feature))


def cube_xyz(state, cube):
    return np.array([value(state, cube, f) for f in ("x", "y", "z")])


def servo(env, state, base_target, grip, steps):
    robot = state.get_object_from_name("robot")
    for _ in range(steps):
        action = np.zeros(18, dtype=np.float32)
        base = np.array([value(state, robot, f) for f in
                         ("pos_base_x", "pos_base_y", "pos_base_rot")])
        joints = np.array([value(state, robot, f"pos_arm_joint{i}")
                           for i in range(1, 8)])
        error = np.asarray(base_target) - base
        error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi
        action[:3] = np.clip(error, -0.1, 0.1)
        action[3:10] = np.clip(HOME - joints, -0.1, 0.1)
        action[10] = grip
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return state


def main():
    env = make_env()
    try:
        trials = 0
        for open_value, close_value in ((0.0, 1.0), (1.0, 0.0)):
            for xoff in np.arange(0.1, 1.0, 0.1):
                for yoff in np.arange(-0.4, 0.41, 0.1):
                    state, _ = env.reset(seed=0, options={"object_count": 1})
                    cube = state.get_object_from_name("cube_0")
                    start = cube_xyz(state, cube)
                    # Offset denotes cube coordinates relative to base.
                    base_target = np.array([start[0] - xoff,
                                            start[1] - yoff, 0.0])
                    state = servo(env, state, base_target, open_value, 60)
                    positioned = cube_xyz(state, cube)
                    state = servo(env, state, base_target, close_value, 25)
                    closed = cube_xyz(state, cube)
                    # A diagonal retreat makes attachment distinguishable from
                    # a base collision along a single axis.
                    retreat = base_target + np.array([-0.25, -0.30, 0.0])
                    state = servo(env, state, retreat, close_value, 45)
                    final = cube_xyz(state, cube)
                    trials += 1
                    approach_move = float(np.linalg.norm(positioned - start))
                    postclose_move = float(np.linalg.norm(final - closed))
                    if approach_move > 0.004 or postclose_move > 0.004:
                        print("MOVE", "trial", trials,
                              "open/close", open_value, close_value,
                              "offset", round(float(xoff), 2), round(float(yoff), 2),
                              "approach", round(approach_move, 4),
                              "postclose", round(postclose_move, 4),
                              "start", np.round(start, 3).tolist(),
                              "positioned", np.round(positioned, 3).tolist(),
                              "closed", np.round(closed, 3).tolist(),
                              "final", np.round(final, 3).tolist(), flush=True)
                    if postclose_move > 0.004:
                        print("FIRST_POSTCLOSE_MOVEMENT; stopping", flush=True)
                        return
            print("POLARITY_COMPLETE", open_value, close_value, "trials", trials,
                  flush=True)
        print("NO_POSTCLOSE_MOVEMENT", "trials", trials, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

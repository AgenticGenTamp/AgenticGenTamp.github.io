"""Test whether the mobile base can displace the target bin."""

import numpy as np

from env_client import make_env


def get(state, obj, feature):
    return float(state.get(obj, feature))


def point(state, obj):
    return np.array([get(state, obj, f) for f in ("x", "y", "z")])


def move_to(env, state, target, grip, steps=100):
    robot = state.get_object_from_name("robot")
    reward = None
    for _ in range(steps):
        now = np.array([get(state, robot, "pos_base_x"),
                        get(state, robot, "pos_base_y")])
        error = np.asarray(target[:2]) - now
        action = np.zeros(18, dtype=np.float32)
        action[:2] = np.clip(error, -0.1, 0.1)
        yaw_error = target[2] - get(state, robot, "pos_base_rot")
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        action[2] = np.clip(yaw_error, -0.1, 0.1)
        action[10] = grip
        state, reward, terminated, truncated, _ = env.step(action)
        if np.linalg.norm(error) < 0.015 or terminated or truncated:
            break
    return state, reward


def trial(grip, direction):
    env = make_env()
    try:
        # Seed 4 puts the bin almost directly behind the central barrier.
        state, _ = env.reset(seed=4, options={"object_count": 1})
        bin_obj = state.get_object_from_name("bin_0")
        cube = state.get_object_from_name("cube_0")
        robot = state.get_object_from_name("robot")
        bin_start = point(state, bin_obj)
        cube_start = point(state, cube)
        bypass_y = 2.65 if bin_start[1] >= 0 else -2.65

        # Cross the barrier's x coordinate away from its y footprint.
        side_yaw = np.pi / 2 if bypass_y > 0 else -np.pi / 2
        route = [(0.0, bypass_y, side_yaw), (1.85, bypass_y, side_yaw)]
        if direction == "forward":
            route.extend([(bin_start[0] - 0.85, bin_start[1], 0.0),
                          (bin_start[0] + 0.65, bin_start[1], 0.0)])
        else:
            # Circle beyond the bin, align behind it, and push toward cube.
            outer_y = bin_start[1] + (1.10 if bin_start[1] <= 0 else -1.10)
            route.extend([(bin_start[0] + 0.85, outer_y, 0.0),
                          (bin_start[0] + 0.85, bin_start[1], 0.0),
                          (max(cube_start[0], 0.4), cube_start[1], 0.0)])

        samples = []
        reward = None
        for waypoint in route:
            state, reward = move_to(env, state, waypoint, grip)
            samples.append((tuple(np.round(point(state, bin_obj), 3)),
                            tuple(np.round([get(state, robot, "pos_base_x"),
                                            get(state, robot, "pos_base_y")], 3)),
                            reward))
        print("TRIAL", direction, "grip", grip,
              "bin_start", np.round(bin_start, 3).tolist(),
              "bin_final", np.round(point(state, bin_obj), 3).tolist(),
              "bin_delta", np.round(point(state, bin_obj) - bin_start, 3).tolist(),
              "cube_delta", np.round(point(state, cube) - cube_start, 3).tolist(),
              "samples", samples, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    for push_direction in ("forward", "reverse"):
        for grip_value in (0.0, 1.0):
            trial(grip_value, push_direction)

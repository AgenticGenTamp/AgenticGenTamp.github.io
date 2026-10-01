"""Calibrate the edge-rake shoulder sweep in four world directions.

The experiment uses an exposed cube from the four-cube pile, approaches it
radially with q1=0, then makes a deliberately short q1 arc.  It reports both
cube and bin displacement so a useful sweep is not confused with pushing a
receptacle.
"""

import math

import numpy as np

from env_client import make_env


Q_EDGE = np.array([0.0, 1.30, math.pi, -1.70, 0.0, 1.0, 0.0])


def xyz(state, name):
    obj = state.get_object_from_name(name)
    return np.array([float(state.get(obj, f)) for f in ("x", "y", "z")])


def robot(state):
    obj = state.get_object_from_name("robot")
    features = (["pos_base_x", "pos_base_y", "pos_base_rot"] +
                [f"pos_arm_joint{i}" for i in range(1, 8)] +
                ["pos_gripper"])
    return np.array([float(state.get(obj, f)) for f in features])


def command(state, base=None, joints=None, grip=0.0, base_limit=0.1):
    r = robot(state)
    action = np.zeros(11, dtype=np.float32)
    if base is not None:
        action[:3] = np.clip(0.8 * (np.asarray(base) - r[:3]),
                                 -base_limit, base_limit)
    if joints is not None:
        action[3:10] = np.clip(0.8 * (np.asarray(joints) - r[3:10]),
                                   -0.1, 0.1)
    action[10] = grip
    return action


def converge(env, state, base=None, joints=None, steps=80):
    for _ in range(steps):
        state, _, _, _, _ = env.step(command(state, base, joints))
    return state


def trial(label, yaw, sign):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 4})
    names = [n for n in state.get_object_names() if n.startswith("cube")]
    bin_names = [n for n in state.get_object_names() if n.startswith("bin_")]
    initial = {n: xyz(state, n) for n in names + bin_names}

    radial = np.array([math.cos(yaw), math.sin(yaw)])
    lateral = np.array([-math.sin(yaw), math.cos(yaw)])
    direction = -sign * lateral
    # Use the outermost cube in the intended sweep direction.  This is the
    # closest available approximation to an isolated cube in the supported
    # four-object minimum.
    target = max(names, key=lambda n: float(np.dot(initial[n][:2], direction)))
    cube_xy = initial[target][:2]
    local_offset = sign * 0.042
    # Preserve the validated yaw-zero geometry exactly: the radial coordinate
    # is relative to the table origin (-1.0 -> -0.8), while only the lateral
    # coordinate follows the chosen cube.  Referencing both coordinates to the
    # cube penetrates 2--4 mm too far and jams shoulder joint 1.
    lateral_coordinate = float(np.dot(cube_xy, lateral) + local_offset)
    staging_xy = -1.00 * radial + lateral_coordinate * lateral
    approach_xy = -0.80 * radial + lateral_coordinate * lateral

    # Keep the folded arm outside the table while translating.  Routing first
    # along the right-hand exterior works for the left and bottom stations
    # used by this calibration.
    # Always reach the west station around the north side, matching the
    # validated experiment.  North/south stations are reached around the
    # right side.
    if staging_xy[0] < -0.70:
        side_y = 1.02
    else:
        side_y = -1.02 if staging_xy[1] < 0.0 else 1.02
    state = converge(env, state, [1.05, side_y, math.pi], None, 55)
    state = converge(env, state, [staging_xy[0], side_y, math.pi], None, 55)
    state = converge(env, state, [staging_xy[0], staging_xy[1], yaw], None, 65)
    state = converge(env, state,
                     [staging_xy[0], staging_xy[1], yaw], Q_EDGE, 120)

    # Advance slowly until any cube moves by 1 mm.
    hit_pose = None
    moved_name = None
    for step in range(55):
        act = command(state, [approach_xy[0], approach_xy[1], yaw],
                      Q_EDGE, base_limit=0.012)
        state, _, _, _, _ = env.step(act)
        movement = {n: np.linalg.norm(xyz(state, n) - initial[n]) for n in names}
        moved_name = max(movement, key=movement.get)
        if movement[moved_name] > 0.001:
            hit_pose = robot(state)[:3].copy()
            break

    if hit_pose is None:
        print(label, "NO_HIT", "target", target,
              "base", np.round(robot(state)[:3], 4).tolist())
        env.close()
        return

    before = {n: xyz(state, n) for n in names + bin_names}
    q_goal = Q_EDGE.copy()
    q_goal[0] = sign * 0.80
    stop_reason = "arc_done"
    arc_steps = 0
    for arc_steps in range(120):
        state, _, _, _, _ = env.step(command(state, hit_pose, q_goal))
        target_delta = xyz(state, target)[:2] - before[target][:2]
        bin_motion = max(np.linalg.norm(xyz(state, n) - before[n])
                         for n in bin_names)
        if np.dot(target_delta, direction) >= 0.080:
            stop_reason = "cube_80mm"
            break

    cube_delta = xyz(state, target) - before[target]
    all_cube_delta = {n: np.round(xyz(state, n) - before[n], 4).tolist()
                      for n in names}
    max_bin = max(np.linalg.norm(xyz(state, n) - before[n]) for n in bin_names)
    print(label, "yaw", round(yaw, 4), "q1_sign", sign,
          "expected_dir", np.round(direction, 3).tolist(),
          "target", target, "first_moved", moved_name,
          "base_minus_cube", np.round(hit_pose[:2] - cube_xy, 4).tolist(),
          "q1", round(float(robot(state)[3]), 4), "steps", arc_steps + 1,
          "stop", stop_reason,
          "target_delta", np.round(cube_delta, 4).tolist(),
          "projection", round(float(np.dot(cube_delta[:2], direction)), 4),
          "max_bin", round(float(max_bin), 4),
          "all_cubes", all_cube_delta)
    env.close()


if __name__ == "__main__":
    # R(yaw) @ (0, -sign): yaw=0 covers world +/-y and yaw=pi/2
    # covers world +/-x.
    trial("world_-y", 0.0, +1.0)
    trial("world_+y", 0.0, -1.0)
    trial("world_+x", math.pi / 2.0, +1.0)
    trial("world_-x", math.pi / 2.0, -1.0)
    trial("world_+x_top", -math.pi / 2.0, -1.0)
    trial("world_-x_top", -math.pi / 2.0, +1.0)

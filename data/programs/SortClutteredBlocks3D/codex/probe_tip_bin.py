"""Tip the red tray, then slide its side/opening toward stationary cube1."""
import sys

import numpy as np

from env_client import make_env
from edge_rake_probe import action_to, robot
from topdown_grasp_probe import get, rob, xyz, act, run


HOME = np.array([0., -.349, np.pi, -2.548, 0., -.873, np.pi / 2])
HIGH = np.array([0., .86, np.pi, -1.70, 0., 1., np.pi / 2])
LOW = HIGH.copy(); LOW[1] = 1.15
PUSH = np.array([0., 1.30, np.pi, -1.70, 0., 1., 0.])


def pose(state, name):
    return get(state, name, ["x", "y", "z", "qw", "qx", "qy", "qz"])


def opening_axis(quat):
    w, x, y, z = quat
    # Third column of quaternion rotation matrix: local tray +Z/opening axis.
    return np.array([2 * (x * z + w * y),
                     2 * (y * z - w * x),
                     1 - 2 * (x * x + y * y)])


def trial(joint, delta, impulse, mode="standard"):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 4})
    b0, c0 = xyz(state, "bin_red").copy(), xyz(state, "cube1").copy()
    initial_direction = c0[:2] - b0[:2]
    initial_direction /= np.linalg.norm(initial_direction)
    # Positive q4 tips the opening along local -X.  In aligned mode choose
    # yaw so local -X points from tray to cube.
    aligned = mode == "aligned"
    yaw = (float(np.arctan2(initial_direction[1], initial_direction[0]) - np.pi)
           if aligned else 0.)
    extension = np.array([np.cos(yaw), np.sin(yaw)])
    base = np.r_[b0[:2] - .91 * extension, yaw]
    state = run(env, state, 35, [1.1, .75, np.pi], HOME, 1)
    if aligned:
        # Folded-arm perimeter routing for the oblique approach.
        angle = float(np.arctan2(base[1], base[0]))
        waypoint = np.array([1.15 * np.cos(angle), 1.15 * np.sin(angle), yaw])
        state = run(env, state, 65, waypoint, HOME, 1)
    else:
        state = run(env, state, 55, [-1.1, .75, np.pi], HOME, 1)
        state = run(env, state, 45, [-1.1, b0[1], 0.], HOME, 1)
    state = run(env, state, 55, base, HOME, 1)
    state = run(env, state, 110, base, HIGH, 1)
    state = run(env, state, 30, base, HIGH, 1)
    state = run(env, state, 45, base, LOW, 1)
    state = run(env, state, 25, base, LOW, 0)

    # Lift just enough to unload the tray, simultaneously bend the wrist and
    # accelerate it toward the cube.  Release while the angular impulse acts.
    goal = base.copy()
    # With q4-positive tipping the opening faces local -X.  A +Y-only impulse
    # should leave the tray directly right of cube1, aligning that opening.
    direction = (np.array([.5, 1.6]) if mode == "ybias"
                 else (np.array([0., 1.]) if mode == "yimpulse"
                       else initial_direction))
    tip_q = LOW.copy(); tip_q[1] = .88; tip_q[joint] += delta
    for k in range(12):
        goal[:2] += impulse * direction
        action = act(state, goal, tip_q, 0 if k < 7 else 1)
        action[:2] = np.clip(action[:2], -.05, .05)
        action[3:10] = np.clip(action[3:10], -.06, .06)
        state, reward, term, trunc, _ = env.step(action)
    state = run(env, state, 90, goal, tip_q, 1)
    tipped_pose = pose(state, "bin_red")
    axis = opening_axis(tipped_pose[3:])
    tilt = float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0., 1.))))
    live_direction = xyz(state, "cube1")[:2] - tipped_pose[:2]
    live_direction /= max(1e-8, np.linalg.norm(live_direction))
    toward = float(np.dot(axis[:2], live_direction))

    # Fold, route behind the tray, and slide it toward cube1 using the same
    # narrow pusher that moves upright trays.  This is attempted regardless of
    # tilt so failed tip trials remain informative controls.
    bp = xyz(state, "bin_red").copy()
    cp = xyz(state, "cube1").copy()
    u = cp[:2] - bp[:2]
    u /= max(1e-8, np.linalg.norm(u))
    yaw = float(np.arctan2(u[1], u[0]))
    current = robot(state)[:3].copy()
    state = run(env, state, 25, current, HOME, 1)
    behind = np.r_[bp[:2] - .98 * u, yaw]
    state = run(env, state, 50, behind, HOME, 1)
    state = run(env, state, 80, behind, PUSH, 1)
    destination = np.r_[cp[:2] - .965 * u, yaw]
    for _ in range(55):
        action = action_to(state, destination, PUSH, 1)
        action[:3] = np.clip(action[:3], -.008, .008)
        state, reward, term, trunc, _ = env.step(action)
    retreat = robot(state)[:3].copy(); retreat[:2] -= .15 * u
    state = run(env, state, 35, retreat, HOME, 1)
    state = run(env, state, 80, retreat, HOME, 1)

    final_bp, final_cp = pose(state, "bin_red"), pose(state, "cube1")
    final_axis = opening_axis(final_bp[3:])
    final_tilt = float(np.degrees(np.arccos(np.clip(abs(final_axis[2]), 0., 1.))))
    print("TIP", joint, delta, impulse, "mode", mode,
          "settled", np.round(tipped_pose, 5).tolist(),
          "tilt", round(tilt, 2), "opening_toward", round(toward, 3),
          "final_bin", np.round(final_bp, 5).tolist(),
          "final_cube", np.round(final_cp, 5).tolist(),
          "final_tilt", round(final_tilt, 2),
          "dxy", round(float(np.linalg.norm(final_bp[:2] - final_cp[:2])), 5),
          "cube_rel", np.round(final_cp[:3] - final_bp[:3], 5).tolist(),
          "reward", reward, "term", term)
    env.close()


if __name__ == "__main__":
    trial(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),
          sys.argv[4] if len(sys.argv) > 4 else "standard")

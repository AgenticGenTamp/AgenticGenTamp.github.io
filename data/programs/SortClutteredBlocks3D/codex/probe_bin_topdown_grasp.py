"""Try grasping the larger red tray, then carry it over stationary cube1."""
import sys

import numpy as np

from env_client import make_env
from topdown_grasp_probe import get, rob, xyz, act, run


HIGH = np.array([0., .86, np.pi, -1.70, 0., 1., np.pi / 2])


def full_pose(state, name):
    return get(state, name, ["x", "y", "z", "qw", "qx", "qy", "qz"])


def trial(xoff, yoff, q2, slow=False):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 4})
    home = rob(state)[3:10].copy()
    bin0 = xyz(state, "bin_red").copy()
    cube0 = xyz(state, "cube1").copy()

    # Keep the arm folded while routing around the table to the bin's left.
    state = run(env, state, 45, [1., .75, np.pi], home, 1)
    state = run(env, state, 55, [-1.1, .75, np.pi], home, 1)
    state = run(env, state, 45, [-1.1, bin0[1] + yoff, 0.], home, 1)
    state = run(env, state, 110, [-1.1, bin0[1] + yoff, 0.], HIGH, 1)

    # The established top-down wrist center is about 0.91 m ahead of base.
    base = np.array([bin0[0] - .91 + xoff, bin0[1] + yoff, 0.])
    state = run(env, state, 30, base, HIGH, 1)
    low = HIGH.copy()
    low[1] = q2
    before = full_pose(state, "bin_red").copy()
    state = run(env, state, 45, base, low, 1)
    down = full_pose(state, "bin_red").copy()
    state = run(env, state, 25, base, low, 0)
    closed = full_pose(state, "bin_red").copy()

    lift = low.copy()
    lift[1] = .82
    peak = closed[2]
    peak_pose = closed.copy()
    lift_steps = 90 if slow else 55
    for lift_step in range(lift_steps):
        lift_goal = lift.copy()
        if slow:
            lift_goal[1] = q2 + (.82 - q2) * ((lift_step + 1) / lift_steps)
        action = act(state, base, lift_goal, 0)
        if slow:
            action[3:10] = np.clip(action[3:10], -.012, .012)
        state, reward, term, trunc, _ = env.step(action)
        now = full_pose(state, "bin_red")
        if now[2] > peak:
            peak, peak_pose = now[2], now.copy()
    lift_end = full_pose(state, "bin_red").copy()
    lifted = lift_end[2] > max(bin0[2], closed[2]) + .02

    carried = False
    carried_pose = None
    dropped = None
    if lifted:
        # Correct from the live tray pose: its XY can drift during the lift,
        # and a one-shot transform based on the earlier peak undershoots.
        destination = rob(state)[:3].copy()
        start_carried = full_pose(state, "bin_red").copy()
        for _ in range(70):
            error = xyz(state, "cube1")[:2] - xyz(state, "bin_red")[:2]
            destination[:2] += np.clip(.7 * error, -.012, .012)
            state, reward, term, trunc, _ = env.step(act(state, destination, lift, 0))
        carried_pose = full_pose(state, "bin_red").copy()
        carried = np.linalg.norm(carried_pose[:2] - peak_pose[:2]) > .03
        # Lower at the shallow, stable q2 used by the cleanest release, while
        # retaining a small live XY correction as the tray meets the table.
        release_q = low.copy()
        release_q[1] = 1.15
        for _ in range(45):
            error = xyz(state, "cube1")[:2] - xyz(state, "bin_red")[:2]
            destination[:2] += np.clip(.4 * error, -.004, .004)
            state, reward, term, trunc, _ = env.step(
                act(state, destination, release_q, 0))
        state = run(env, state, 25, destination, release_q, 1)
        state = run(env, state, 25, destination, HIGH, 1)
        # Withdraw so neither a finger nor the palm can prop up the tray.
        destination[0] -= .18
        state = run(env, state, 45, destination, HIGH, 1)
        for _ in range(50):
            state, reward, term, trunc, _ = env.step(act(state, destination, HIGH, 1))
        dropped = full_pose(state, "bin_red").copy()

    print("BIN_GRASP", "xoff", xoff, "yoff", yoff, "q2", q2,
          "before", np.round(before, 5).tolist(),
          "down", np.round(down, 5).tolist(),
          "closed", np.round(closed, 5).tolist(),
          "peak", round(float(peak), 5),
          "lift_end", np.round(lift_end, 5).tolist(), "slow", slow,
          "lifted", lifted, "carried", carried,
          "carried_pose", None if carried_pose is None else np.round(carried_pose, 5).tolist(),
          "drop", None if dropped is None else np.round(dropped, 5).tolist(),
          "cube0", np.round(cube0, 5).tolist(),
          "cube_final", np.round(xyz(state, "cube1"), 5).tolist(),
          "reward", reward, "term", term)
    env.close()


if __name__ == "__main__":
    trial(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),
          len(sys.argv) > 4 and sys.argv[4] == "slow")

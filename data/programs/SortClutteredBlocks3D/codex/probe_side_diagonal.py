"""Search table-height side grasps after mechanically isolating one cube.

The known edge pose can touch a cube, so each trial scans the base only until
first contact, closes in stages, and tests whether the cube follows a retreat.
"""
from env_client import make_env
import numpy as np

from topdown_grasp_probe import act, rob, run, xyz


EDGE = np.array([0.0, 1.30, np.pi, -1.70, 0.0, 1.0, 0.0])
HOME_HIGH = np.array([0.0, 0.90, np.pi, -1.70, 0.0, 1.0, 0.0])


def isolate(env, state, home):
    """Use the repeatable q1 tangent sweep to pull cube1 clear of the pile."""
    cube = "cube1"
    state = run(env, state, 45, [1.0, 0.8, np.pi], home, 1)
    state = run(env, state, 55, [-1.0, 0.8, np.pi], home, 1)
    state = run(env, state, 45, [-1.0, 0.8, 0.0], home, 1)
    y = float(xyz(state, cube)[1] + 0.042)
    state = run(env, state, 45, [-1.0, y, 0.0], home, 1)
    state = run(env, state, 130, [-1.0, y, 0.0], EDGE, 1)
    baseline = {n: xyz(state, n).copy() for n in state.get_object_names()
                if n.startswith("cube")}
    hit = None
    for _ in range(45):
        command = act(state, [-0.78, y, 0.0], EDGE, 1)
        command[0] = min(command[0], 0.012)
        state, *_ = env.step(command)
        if max(np.linalg.norm(xyz(state, n) - p)
               for n, p in baseline.items()) > 0.001:
            hit = rob(state)[:3].copy()
            break
    if hit is None:
        hit = rob(state)[:3].copy()
    sweep = EDGE.copy(); sweep[0] = 0.50
    for _ in range(80):
        state, *_ = env.step(act(state, hit, sweep, 1))
        if xyz(state, cube)[1] < -0.085:
            break
    return state, cube


def trial(q7, y_offset):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 4})
    home = rob(state)[3:10].copy()
    state, cube = isolate(env, state, home)
    isolated = xyz(state, cube).copy()

    # Withdraw, rotate the terminal wrist, and approach the isolated cube.
    pose = EDGE.copy(); pose[6] = q7
    high = HOME_HIGH.copy(); high[6] = q7
    y = float(isolated[1] + y_offset)
    state = run(env, state, 35, [-1.10, y, 0.0], high, 1)
    state = run(env, state, 80, [-1.10, y, 0.0], pose, 1)
    before = xyz(state, cube).copy()
    contact_base = None
    for _ in range(50):
        command = act(state, [-0.68, y, 0.0], pose, 1)
        command[0] = min(command[0], 0.008)
        state, *_ = env.step(command)
        if np.linalg.norm(xyz(state, cube) - before) > 0.001:
            contact_base = rob(state)[:3].copy()
            break
    if contact_base is None:
        contact_base = rob(state)[:3].copy()
    contact = xyz(state, cube).copy()

    # Let each intermediate finger target settle instead of snapping shut.
    for grip in (0.75, 0.50, 0.25, 0.0):
        state = run(env, state, 10, contact_base, pose, grip)
    closed = xyz(state, cube).copy()
    peak_z = closed[2]
    retreat = contact_base.copy(); retreat[0] -= 0.18
    for _ in range(55):
        state, *_ = env.step(act(state, retreat, high, 0))
        peak_z = max(peak_z, xyz(state, cube)[2])
    final = xyz(state, cube).copy()
    print("SIDE", round(q7, 3), "yoff", y_offset,
          "isolated", np.round(isolated, 4).tolist(),
          "base", np.round(contact_base, 4).tolist(),
          "contact_d", np.round(contact - before, 4).tolist(),
          "close_d", np.round(closed - contact, 4).tolist(),
          "retreat_d", np.round(final - closed, 4).tolist(),
          "peak_z", round(float(peak_z), 4))
    env.close()


if __name__ == "__main__":
    for params in ((-0.8, 0.0), (-0.4, 0.0), (0.0, -0.018),
                   (0.0, 0.0), (0.0, 0.018), (0.4, 0.0), (0.8, 0.0)):
        trial(*params)

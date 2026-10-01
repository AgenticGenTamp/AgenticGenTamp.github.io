"""Search side-contact joint impulses for an upward scoop component.

This intentionally differs from the top-down pinch probes: cube1 is exposed
with a fingertip rake, stopped before the red tray, and a single arm joint is
snapped while the closed/open hand remains in lateral contact.
"""
import sys

import numpy as np

from env_client import make_env
from topdown_grasp_probe import act, rob, run, xyz


EDGE = np.array([0.0, 1.3, np.pi, -1.7, 0.0, 1.0, 0.0])


def trial(joint, delta, grip, grip_after=None, base_delta=None, repeats=1,
          continuous_steps=0, pulse_period=0, release_steps=0):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 4})
    home = rob(state)[3:10]
    cube = "cube1"
    names = [n for n in state.get_object_names() if n.startswith("cube")]

    state = run(env, state, 45, [1, .8, np.pi], home, grip)
    state = run(env, state, 55, [-1, .8, np.pi], home, grip)
    state = run(env, state, 45, [-1, .8, 0], home, grip)
    ybase = float(xyz(state, cube)[1] + .042)
    state = run(env, state, 45, [-1, ybase, 0], home, grip)
    state = run(env, state, 130, [-1, ybase, 0], EDGE, grip)
    baseline = {n: xyz(state, n).copy() for n in names}
    hit = None
    for _ in range(45):
        command = act(state, [-.8, ybase, 0], EDGE, grip)
        command[0] = min(command[0], .012)
        state, *_ = env.step(command)
        if max(np.linalg.norm(xyz(state, n) - baseline[n]) for n in names) > .001:
            hit = rob(state)[:3].copy()
            break
    if hit is None:
        print("NO_CONTACT")
        env.close()
        return

    sweep = EDGE.copy(); sweep[0] = .45
    # Stop safely before the tray wall at y=-0.10.
    for _ in range(50):
        state, *_ = env.step(act(state, hit, sweep, grip))
        if xyz(state, cube)[1] < -.058:
            break
    before = xyz(state, cube).copy()
    bin_before = xyz(state, "bin_red").copy()
    q = rob(state)[3:10].copy()
    goal = q.copy(); goal[joint - 1] += delta
    final_grip = grip if grip_after is None else grip_after
    peak_z = before[2]
    samples = []
    phases = []
    for phase in range(repeats if not continuous_steps else continuous_steps):
        move_base = hit.copy()
        if base_delta is not None:
            if continuous_steps:
                move_base = rob(state)[:3].copy()
                move_base[:2] += base_delta
            else:
                move_base[:2] += (phase + 1) * base_delta
        phase_start = xyz(state, cube).copy()
        for step in range(1 if continuous_steps else 12):
            commanded_grip = final_grip
            if pulse_period:
                commanded_grip = (1.0 if phase % pulse_period < pulse_period // 2
                                    else 0.0)
            state, *_ = env.step(act(state, move_base, goal, commanded_grip))
            pos = xyz(state, cube).copy()
            vel = np.array([state.get(state.get_object_from_name(cube), f)
                            for f in ("vx", "vy", "vz")])
            peak_z = max(peak_z, pos[2])
            if phase == 0 and step < 4:
                samples.append((np.round(pos - before, 5).tolist(),
                                np.round(vel, 4).tolist()))
        phases.append(np.round(xyz(state, cube) - phase_start, 5).tolist())
    before_release = xyz(state, cube).copy()
    if release_steps:
        hold_base = rob(state)[:3].copy()
        for _ in range(release_steps):
            state, *_ = env.step(act(state, hold_base, goal, 1.0))
    release_move = xyz(state, cube) - before_release
    print("SCOOP", joint, delta, "grip", grip, "to", final_grip,
          "q0", np.round(q, 3).tolist(),
          "dpos", np.round(xyz(state, cube) - before, 5).tolist(),
          "peak_dz", round(float(peak_z - before[2]), 5),
          "bin_move", round(float(np.linalg.norm(xyz(state, "bin_red") - bin_before)), 5),
          "phases", phases,
          "release_move", np.round(release_move, 5).tolist(),
          "samples", samples)
    env.close()


if __name__ == "__main__":
    trial(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),
          float(sys.argv[4]) if len(sys.argv) > 4 else None,
          np.array([float(sys.argv[5]), float(sys.argv[6])])
          if len(sys.argv) > 6 else None,
          int(sys.argv[7]) if len(sys.argv) > 7 else 1,
          int(sys.argv[8]) if len(sys.argv) > 8 else 0,
          int(sys.argv[9]) if len(sys.argv) > 9 else 0,
          int(sys.argv[10]) if len(sys.argv) > 10 else 0)

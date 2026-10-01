"""Test a short upward fingertip impulse before a cube reaches its bin wall."""
import sys
import numpy as np

from env_client import make_env
from edge_rake_probe import Q, action_to, cubes, get, robot, settle_to


def run(lift_q2, base_impulse=False):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 4})
    initial = cubes(state)
    name = "cube1"
    initial_bin = get(state, "bin_red", ["x", "y", "z"]).copy()
    state = settle_to(env, state, [.99, .80, np.pi], None, 45, 0)
    state = settle_to(env, state, [-1., .80, np.pi], None, 55, 0)
    state = settle_to(env, state, [-1., .80, 0.], None, 45, 0)
    ybase = float(initial[name][1] + .042)
    state = settle_to(env, state, [-1., ybase, 0.], None, 45, 0)
    state = settle_to(env, state, [-1., ybase, 0.], Q, 130, 0)
    hit = None
    for _ in range(50):
        action = action_to(state, [-.80, ybase, 0.], Q, 0)
        action[0] = min(action[0], .012)
        state, _, _, _, _ = env.step(action)
        if max(np.linalg.norm(v - initial[n]) for n, v in cubes(state).items()) > .001:
            hit = robot(state)[:3].copy()
            break
    if hit is None:
        print("NO_HIT")
        env.close()
        return

    sweep = Q.copy()
    sweep[0] = .8
    # Establish controlled motion but launch well before touching the bin.
    for _ in range(24 if base_impulse else 38):
        state, _, _, _, _ = env.step(action_to(state, hit, sweep, 0))
    print("PRE", lift_q2, np.round(cubes(state)[name], 5),
          "v", np.round(get(state, name, ["vx", "vy", "vz"]), 5),
          "bin", np.round(get(state, "bin_red", ["x", "y", "z"]), 5))

    if base_impulse:
        # A two-step diagonal chassis snap tests whether impact momentum can
        # carry the cube over the wall instead of quasistatically moving it.
        for step in range(2):
            action = action_to(state, hit, sweep, 0)
            action[0], action[1] = -.1, -.1
            state, reward, term, trunc, _ = env.step(action)
            print("IMPACT", step, reward, np.round(cubes(state)[name], 5),
                  np.round(get(state, name, ["vx", "vy", "vz"]), 4))
        for step in range(90):
            action = action_to(state, hit + np.array([.2, .2, 0.]), Q, 0)
            state, reward, term, trunc, _ = env.step(action)
            cube = cubes(state)[name]
            if step < 15 or step % 10 == 0 or term:
                print("COAST", step, reward, term, np.round(cube, 5),
                      np.round(get(state, name, ["vx", "vy", "vz"]), 4),
                      "bin", np.round(get(state, "bin_red", ["x", "y", "z"]), 4))
        env.close()
        return

    lift = sweep.copy()
    lift[1] = lift_q2
    # Keep sweeping laterally during the initial upward snap, then reverse q1
    # to release the cube from the fingertip.
    for step in range(90):
        if step == 14:
            lift[0] = -.25
        state, reward, term, trunc, _ = env.step(action_to(state, hit, lift, 0))
        cube = cubes(state)[name]
        if step < 25 or step % 10 == 0 or reward != -1.0 or term:
            print("FLIGHT", step, "r", reward, "term", term,
                  "cube", np.round(cube, 5),
                  "v", np.round(get(state, name, ["vx", "vy", "vz"]), 4),
                  "d", round(float(np.linalg.norm(cube - initial_bin)), 4),
                  "bin", np.round(get(state, "bin_red", ["x", "y", "z"]), 4))
        if term or trunc:
            break
    env.close()


if __name__ == "__main__":
    run(float(sys.argv[1]) if len(sys.argv) > 1 else .4,
        len(sys.argv) > 2 and sys.argv[2] == "impulse")

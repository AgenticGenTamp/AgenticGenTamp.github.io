"""Read-only diagnostic runner for the current submitted policy."""
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def vals(state, obj, names):
    return np.array([float(state.get(obj, n)) for n in names])


def run(seed, limit=330):
    env = make_env()
    state, info = env.reset(seed=seed)
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    robot = state.get_object_from_name("robot")
    initial = {r.name: vals(state, r, ("x", "y", "z")) for r in policy.rods}
    peak = {r.name: 0.0 for r in policy.rods}
    total = 0.0
    print("SEED", seed, "n", len(policy.rods), "slots", np.round(policy.slot_y, 3),
          "initial", {k: np.round(v, 3) for k, v in initial.items()})
    for step in range(limit):
        action = policy.get_action(state)
        state, reward, term, trunc, info = env.step(action)
        total += reward
        for r in policy.rods:
            peak[r.name] = max(peak[r.name], float(np.linalg.norm(
                vals(state, r, ("x", "y", "z")) - initial[r.name])))
        if step in (0, 19, 39, 79, 99, 109, 129, 154, 159, 179,
                    239, 259, 269, 289, 314, 319) or reward > 0:
            base = vals(state, robot, ("pos_base_x", "pos_base_y", "pos_base_rot"))
            q = vals(state, robot, tuple(f"pos_arm_joint{i}" for i in range(1, 8)))
            target = policy.home if step < 20 else policy.pick_q
            qerr = (target - q + np.pi) % (2*np.pi) - np.pi
            rp = vals(state, policy.rods[min(step//160, len(policy.rods)-1)], ("x", "y", "z"))
            grip = float(state.get(robot, "pos_gripper"))
            print(step+1, "r", round(reward, 3), "base", np.round(base, 3),
                  "qerr", round(float(np.max(np.abs(qerr))), 3),
                  "grip", round(grip, 3), "rod", np.round(rp, 3))
        if term or trunc:
            break
    print("END", step+1, round(total, 3), term, trunc, "peak", peak, "info", info)
    env.close()


if __name__ == "__main__":
    run(0)
    run(3)

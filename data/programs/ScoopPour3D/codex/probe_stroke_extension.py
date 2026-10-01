"""Probe sweep-axis schedules without modifying the submitted policy."""

import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def xy(state, name):
    obj = state.get_object_from_name(name)
    return np.array([state.get(obj, f) for f in ("x", "y")], float)


def alter(action, step, mode):
    """Apply one candidate around the productive sweep (steps 194 onward)."""
    if mode == "pre10" and 184 <= step <= 193:
        action[5] = 0.1
    elif mode == "pre5" and 189 <= step <= 193:
        action[5] = 0.1
    elif mode == "pre8" and 186 <= step <= 193:
        action[5] = 0.1
    elif mode == "pre20" and 174 <= step <= 193:
        action[5] = 0.1
    elif mode == "pre30" and 164 <= step <= 193:
        action[5] = 0.1
    elif mode == "rev10" and 226 <= step <= 235:
        action[5] = 0.1
    elif mode == "rev20" and 221 <= step <= 240:
        action[5] = 0.1
    elif mode == "switch3" and 226 <= step <= 320:
        action[5], action[3] = 0.0, -0.1
    elif mode == "switch3_216" and 216 <= step <= 320:
        action[5], action[3] = 0.0, -0.1
    elif mode == "switch3_236" and 236 <= step <= 320:
        action[5], action[3] = 0.0, -0.1
    elif mode == "switch3_246" and 246 <= step <= 320:
        action[5], action[3] = 0.0, -0.1
    elif mode == "switch3_242" and 242 <= step <= 320:
        action[5], action[3] = 0.0, -0.1
    elif mode == "switch3_244" and 244 <= step <= 320:
        action[5], action[3] = 0.0, -0.1
    elif mode == "switch3_248" and 248 <= step <= 320:
        action[5], action[3] = 0.0, -0.1
    elif mode == "switch3_250" and 250 <= step <= 320:
        action[5], action[3] = 0.0, -0.1
    elif mode == "switch3_256" and 256 <= step <= 320:
        action[5], action[3] = 0.0, -0.1
    elif mode == "switch3_266" and 266 <= step <= 320:
        action[5], action[3] = 0.0, -0.1
    elif mode == "switch4_246" and 246 <= step <= 320:
        action[5], action[4] = 0.0, -0.1
    elif mode == "switch6_246" and 246 <= step <= 320:
        action[5], action[6] = 0.0, -0.1
    elif mode == "switch8_246" and 246 <= step <= 320:
        action[5], action[8] = 0.0, -0.1
    elif mode == "switch9_246" and 246 <= step <= 320:
        action[5], action[9] = 0.0, -0.1
    elif mode == "switch3_pos" and 226 <= step <= 320:
        action[5], action[3] = 0.0, 0.1
    elif mode == "pre5_switch3":
        if 189 <= step <= 193:
            action[5] = 0.1
        elif 226 <= step <= 320:
            action[5], action[3] = 0.0, -0.1
    elif mode == "pre10_switch3":
        if 184 <= step <= 193:
            action[5] = 0.1
        elif 226 <= step <= 320:
            action[5], action[3] = 0.0, -0.1
    elif mode == "switch7" and 226 <= step <= 320:
        action[5], action[7] = 0.0, -0.1
    elif mode == "alternate":
        if 224 <= step <= 243:
            action[5] = 0.1
        elif 244 <= step <= 290:
            action[5] = -0.1
    return action


def run(seed, count, mode):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": count})
    names = sorted(n for n in state.get_object_names() if n.startswith("cube_"))
    initial = np.array([xy(state, n) for n in names])
    desired = initial + xy(state, "bin_green_0") - xy(state, "bin_yellow_0")
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    reward = 0.0
    done = False
    for step in range(1, 321):
        action = alter(policy.get_action(state), step, mode)
        state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        if done:
            break
    final = np.array([xy(state, n) for n in names])
    err = np.linalg.norm(final - desired, axis=1)
    disp = (final - initial).mean(axis=0)
    print(seed, mode, "disp", np.round(disp, 4), "err", np.round(
        [err.min(), err.mean(), err.max()], 4), "within", int((err < .05).sum()),
        "reward", reward, "done", done)
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("modes", nargs="*", default=["base", "pre10", "pre20",
                        "pre30", "rev10", "rev20", "switch3", "switch7",
                        "alternate"])
    args = parser.parse_args()
    for mode in args.modes:
        run(args.seed, args.count, mode)


if __name__ == "__main__":
    main()

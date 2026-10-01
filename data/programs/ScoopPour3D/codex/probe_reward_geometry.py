"""Log reward against cube distances to both bins along the submitted policy."""

import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def xyz(state, name):
    obj = state.get_object_from_name(name)
    return np.array([state.get(obj, key) for key in ("x", "y", "z")], float)


def summarize(state, initial, reward, step):
    names = [n for n in state.get_object_names() if n.startswith("cube_")]
    points = np.array([xyz(state, n) for n in names])
    green = xyz(state, "bin_green_0")
    yellow = xyz(state, "bin_yellow_0")
    dg = np.linalg.norm(points - green, axis=1)
    dgxy = np.linalg.norm(points[:, :2] - green[:2], axis=1)
    dyxy = np.linalg.norm(points[:, :2] - yellow[:2], axis=1)
    # A translated-position hypothesis: each target preserves a cube's
    # initial offset inside yellow, but shifts that offset to green.
    translated = initial + (xyz0_green - xyz0_yellow)
    dt = np.linalg.norm(points - translated, axis=1)
    print(step, "r", reward, "mean", np.round(points.mean(0), 4),
          "green", np.round(green, 4), "yellow", np.round(yellow, 4),
          "g3", np.round([dg.min(), dg.mean(), dg.max()], 4),
          "gxy", np.round([dgxy.min(), dgxy.mean(), dgxy.max()], 4),
          "yxy", np.round([dyxy.min(), dyxy.mean(), dyxy.max()], 4),
          "translated", np.round([dt.min(), dt.mean(), dt.max()], 4),
          "near5/10", int((dgxy < .05).sum()), int((dgxy < .10).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--correct-x", type=float, default=0.0,
                        help="constant command after policy steps")
    parser.add_argument("--correct-index", type=int, default=0)
    parser.add_argument("--correct-index2", type=int, default=-1)
    parser.add_argument("--correct-value2", type=float, default=0.0)
    parser.add_argument("--correct2-steps", type=int, default=-1)
    parser.add_argument("--correct-index3", type=int, default=-1)
    parser.add_argument("--correct-value3", type=float, default=0.0)
    parser.add_argument("--correct-index4", type=int, default=-1)
    parser.add_argument("--correct-value4", type=float, default=0.0)
    parser.add_argument("--correct-steps", type=int, default=0)
    args = parser.parse_args()
    env = make_env()
    global xyz0_yellow, xyz0_green
    state, info = env.reset(seed=args.seed, options={"object_count": args.count})
    names = [n for n in state.get_object_names() if n.startswith("cube_")]
    initial = np.array([xyz(state, n) for n in names])
    xyz0_yellow = xyz(state, "bin_yellow_0")
    xyz0_green = xyz(state, "bin_green_0")
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    summarize(state, initial, 0.0, 0)
    previous_reward = None
    for step in range(1, args.steps + 1):
        state, reward, term, trunc, _ = env.step(policy.get_action(state))
        if (reward != previous_reward or step % 25 == 0 or term or trunc):
            summarize(state, initial, reward, step)
        previous_reward = reward
        if term or trunc:
            break
    if not (term or trunc):
        for extra in range(1, args.correct_steps + 1):
            action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            action[args.correct_index] = args.correct_x
            if args.correct_index2 >= 0:
                if args.correct2_steps < 0 or extra <= args.correct2_steps:
                    action[args.correct_index2] = args.correct_value2
            if args.correct_index3 >= 0:
                action[args.correct_index3] = args.correct_value3
            if args.correct_index4 >= 0:
                action[args.correct_index4] = args.correct_value4
            state, reward, term, trunc, _ = env.step(action)
            if reward != previous_reward or extra % 5 == 0 or term or trunc:
                summarize(state, initial, reward, args.steps + extra)
            previous_reward = reward
            if term or trunc:
                break
    env.close()


if __name__ == "__main__":
    main()

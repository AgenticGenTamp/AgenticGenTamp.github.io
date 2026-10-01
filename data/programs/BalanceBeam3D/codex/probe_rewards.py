"""Small black-box probe for BalanceBeam3D resets and rewards."""
import numpy as np
import sys

from env_client import make_env


OBJECT_STARTS = (0, 38, 54, 70)


def summarize(obs):
    xyz = [np.round(obs[i:i + 3], 4).tolist() for i in OBJECT_STARTS]
    bbs = [np.round(obs[i + 13:i + 16], 4).tolist() for i in OBJECT_STARTS]
    robot = np.round(obs[16:27], 4).tolist()
    return xyz, bbs, robot


def neutral(obs):
    action = np.zeros(11, dtype=np.float32)
    action[10] = np.clip(obs[26], 0.0, 1.0)
    return action


def reset_survey():
    print("RESET SURVEY")
    for seed in range(10):
        env = make_env()
        obs, info = env.reset(seed=seed)
        xyz, bbs, robot = summarize(obs)
        print(seed, "xyz", xyz, "bb", bbs, "robot", robot, "info", info,
              "max", env.max_steps)
        env.close()


def rollout(label, seed, action_fn, steps=20):
    env = make_env()
    obs, info0 = env.reset(seed=seed)
    initial = obs.copy()
    rewards = []
    infos = []
    end = (False, False)
    for t in range(steps):
        action = action_fn(obs, t).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        if info and info not in infos:
            infos.append(info)
        end = terminated, truncated
        if terminated or truncated:
            break
    changed = np.where(np.abs(obs - initial) > 1e-4)[0]
    print(label, "n", len(rewards), "rewards", rewards, "sum", sum(rewards),
          "term", end, "changed", changed.tolist(), "info0", info0,
          "infos", infos, "start/end xyz", summarize(initial)[0], summarize(obs)[0],
          "robot_end", summarize(obs)[2])
    env.close()


def command(axis, value, grip=None):
    def fn(obs, _t):
        a = neutral(obs)
        a[axis] = value
        if grip is not None:
            a[10] = grip
        return a
    return fn


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "resets"):
        reset_survey()
    if mode in ("all", "actions"):
        rollout("neutral seed0", 0, lambda obs, t: neutral(obs), 40)
        rollout("neutral seed7", 7, lambda obs, t: neutral(obs), 20)
        for axis, name in [(0, "base+x"), (1, "base+y"), (2, "base+yaw"),
                           (3, "joint1"), (9, "joint7")]:
            rollout(name, 0, command(axis, 0.1), 10)
        rollout("grip-open", 0, command(0, 0.0, 1.0), 10)
        rollout("grip-close", 0, command(0, 0.0, 0.0), 10)
    if mode == "collisions":
        rollout("base+x through seed7 objects", 7, command(0, 0.1), 35)
        rollout("base+x through seed2 objects", 2, command(0, 0.1), 35)
    if mode == "random":
        rng = np.random.default_rng(123)

        def random_action(obs, _t):
            a = rng.uniform(-0.1, 0.1, 11).astype(np.float32)
            a[10] = rng.uniform(0.0, 1.0)
            return a

        rollout("random controls seed0", 0, random_action, 200)

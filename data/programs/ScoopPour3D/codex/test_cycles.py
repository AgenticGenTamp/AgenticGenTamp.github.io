"""Repeatedly realign the arm and paddle the scoop toward the source bin."""

import numpy as np
from env_client import make_env


QFS = [f"pos_arm_joint{i}" for i in range(1, 8)]


def get(s, name, fs):
    o = s.get_object_from_name(name)
    return np.array([s.get(o, f) for f in fs], float)


def step(env, s, a):
    return env.step(np.asarray(a, np.float32))[0]


def main():
    env = make_env()
    s, _ = env.reset(seed=0)
    home = get(s, "robot", QFS)
    base0 = get(s, "robot", ("pos_base_x", "pos_base_y"))
    rel = get(s, "scoop_0", ("x", "y")) - base0
    for cycle in range(5):
        # Return arm home before moving the base, avoiding a low-arm collision.
        for _ in range(35):
            a = np.zeros(11); a[10] = 1
            q = get(s, "robot", QFS)
            a[3:10] = np.clip(0.8 * (home - q), -.1, .1)
            s = step(env, s, a)
        # Align the initial contact point over the new scoop XY.
        for _ in range(25):
            a = np.zeros(11); a[10] = 1
            scoop = get(s, "scoop_0", ("x", "y"))
            base = get(s, "robot", ("pos_base_x", "pos_base_y"))
            a[:2] = np.clip(0.5 * (scoop - rel - base), -.1, .1)
            s = step(env, s, a)
        print("aligned", cycle, np.round(get(s, "robot", ("pos_base_x", "pos_base_y")), 3),
              np.round(get(s, "scoop_0", ("x", "y", "z")), 3))
        for _ in range(32):
            a = np.zeros(11); a[10] = 1; a[4] = .1; a[6] = .06
            s = step(env, s, a)
        for _ in range(5):
            a = np.zeros(11); a[10] = 1; a[8] = .1
            s = step(env, s, a)
        for _ in range(10):
            a = np.zeros(11); a[10] = 1; a[0] = .02; a[1] = -.05
            s = step(env, s, a)
        cubes = np.array([get(s, n, ("x", "y", "z")) for n in s.get_object_names() if n.startswith("cube_")])
        print(cycle, "base", np.round(get(s, "robot", ("pos_base_x", "pos_base_y")), 3),
              "scoop", np.round(get(s, "scoop_0", ("x", "y", "z")), 3),
              "cubes", np.round(cubes.mean(axis=0), 3))
    env.close()


if __name__ == "__main__":
    main()

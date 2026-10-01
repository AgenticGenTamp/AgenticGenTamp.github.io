import sys
import numpy as np
from env_client import make_env


def summary(env, state):
    out = {}
    for typ in env.observation_space.types:
        objs = state.get_objects(typ)
        if not objs:
            continue
        out[typ.name] = {}
        for obj in objs:
            name = getattr(obj, "name", str(obj))
            vals = {}
            for feat in env.observation_space.type_features[typ]:
                try:
                    vals[feat] = float(state.get(obj, feat))
                except Exception:
                    pass
            out[typ.name][name] = vals
    return out


def flat_relevant(env, state):
    s = summary(env, state)
    robot = next(iter(s.get("mujoco_tidybot_robot", {}).values()), {})
    mov = s.get("mujoco_movable_object", {})
    return robot, mov


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    env = make_env()
    state, info = env.reset(seed=seed)
    print("space", env.action_space.low, env.action_space.high, "max", env.max_steps)
    print("names", state.get_object_names())
    print("info", info)
    r0, m0 = flat_relevant(env, state)
    print("robot0", r0)
    print("mov0", m0)
    dims = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for dim in dims:
        state, info = env.reset(seed=seed)
        rb, mb = flat_relevant(env, state)
        total = 0.0
        for _ in range(10):
            a = np.zeros(11, dtype=np.float32)
            a[dim] = 1.0 if dim == 10 else 0.1
            state, reward, term, trunc, info = env.step(a)
            total += reward
        ra, ma = flat_relevant(env, state)
        rd = {k: round(ra.get(k, 0) - rb.get(k, 0), 4) for k in ra}
        md = {}
        for n, vals in ma.items():
            if n in mb:
                d = [vals.get(c, 0)-mb[n].get(c, 0) for c in ("x", "y", "z")]
                if max(map(abs, d)) > 1e-5:
                    md[n] = [round(x, 4) for x in d]
        print("dim", dim, "reward", round(total, 4), "robot_delta", rd, "mov_delta", md,
              "last_info", info, "done", term, trunc)
    env.close()


if __name__ == "__main__":
    main()

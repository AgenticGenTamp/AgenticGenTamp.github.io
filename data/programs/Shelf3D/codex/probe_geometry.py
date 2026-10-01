from env_client import make_env


def one(seed, count=None):
    env = make_env()
    options = None if count is None else {"object_count": count}
    try:
        obs, info = env.reset(seed=seed, options=options)
        print("RESET", seed, count, "max", env.max_steps)
        print("info", repr(info))
        print("names", obs.get_object_names())
        for name in obs.get_object_names():
            obj = obs.get_object_from_name(name)
            typ = getattr(obj, "type", None)
            if typ is None:
                typ = getattr(obj, "type_name", None)
            print(name, "type", typ, "repr", repr(obj))
            # Ask all declared features and retain successful reads.
            vals = {}
            for tname, feats in env.observation_space.type_features.items():
                for feat in feats:
                    try:
                        vals[feat] = float(obs.get(obj, feat))
                    except Exception:
                        pass
            print(" vals", vals)
    except Exception as exc:
        print("ERROR", seed, count, type(exc).__name__, exc)
    finally:
        env.close()


if __name__ == "__main__":
    for args in [(0, None), (1, None), (2, None), (42, None),
                 (0, 1), (0, 2), (0, 5), (9, 8)]:
        one(*args)

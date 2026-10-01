"""Small black-box state dumper; not imported by the submitted approach."""

from env_client import make_env


def main():
    for seed in range(3):
        env = make_env()
        state, info = env.reset(seed=seed)
        print("seed", seed, "info", info, "max", env.max_steps)
        for name in state.get_object_names():
            obj = state.get_object_from_name(name)
            typ = getattr(obj, "type", None)
            print(name, typ, obj)
            for candidate in env.observation_space.types:
                try:
                    vals = [state.get(obj, f) for f in candidate.features]
                except Exception:
                    continue
                print(" ", candidate.name, dict(zip(candidate.features, vals)))
                break
        env.close()


if __name__ == "__main__":
    main()

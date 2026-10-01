"""Print compact initial-state geometry for Packing3DEnv exploration."""

import argparse

from env_client import make_env


def describe(state):
    rows = []
    for name in sorted(state.get_object_names()):
        obj = state.get_object_from_name(name)
        features = state.type_features[obj.type]
        values = {feature: state.get(obj, feature) for feature in features}
        rows.append((name, obj.type.name, values))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--object-count", type=int)
    args = parser.parse_args()
    for seed in args.seeds:
        env = make_env()
        try:
            options = ({"object_count": args.object_count}
                       if args.object_count is not None else None)
            state, info = env.reset(seed=seed, options=options)
            print(f"SEED {seed} options={options} info={info}")
            for name, type_name, values in describe(state):
                compact = " ".join(f"{key}={value:.6g}" for key, value in values.items())
                print(f"  {name:8s} {type_name:22s} {compact}")
        finally:
            env.close()


if __name__ == "__main__":
    main()

"""Concise reset-state survey for Transport3DEnv (exploration only)."""

from env_client import make_env


def survey(seed, object_count=None):
    env = make_env()
    try:
        options = None if object_count is None else {"object_count": object_count}
        state, info = env.reset(seed=seed, options=options)
        print(f"\nseed={seed} requested_count={object_count} info={info}")
        print("names", sorted(state.get_object_names()))
        for name in sorted(state.get_object_names()):
            obj = state.get_object_from_name(name)
            features = state.type_features[obj.type]
            vals = {f: state.get(obj, f) for f in features}
            print(name, obj.type.name, vals)
        print("max_steps", env.max_steps)
        print("action", env.action_space.shape, env.action_space.dtype,
              env.action_space.low.tolist(), env.action_space.high.tolist())
        print("types", [(t.name, env.observation_space.type_features[t])
                        for t in env.observation_space.types])
    finally:
        env.close()


if __name__ == "__main__":
    for args in [(0, None), (1, None), (2, None), (10, 1), (11, 2),
                 (12, 3), (13, 5), (14, 8)]:
        survey(*args)

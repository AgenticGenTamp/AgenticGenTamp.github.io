"""Read-only probe for reset metadata and object-centric state API."""

from env_client import make_env


def compact(value, limit=1000):
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


for count in (1, 2, 4, 6):
    for seed in (0, 7):
        env = make_env()
        try:
            state, info = env.reset(seed=seed, options={"object_count": count})
        except TypeError:
            # Server-specific count selection is normally unavailable through
            # reset; still inspect naturally sampled episodes.
            state, info = env.reset(seed=seed)
        print("CASE", count, seed)
        print("info", compact(info, 4000))
        print("state type", type(state), "dict", compact(getattr(state, "__dict__", None), 4000))
        print("state public", [x for x in dir(state) if not x.startswith("_")])
        print("names", state.get_object_names())
        for name in state.get_object_names():
            obj = state.get_object_from_name(name)
            print("obj", name, type(obj), compact(getattr(obj, "__dict__", None), 1000))
        print("space dict", compact(getattr(env.observation_space, "__dict__", None), 4000))
        env.close()

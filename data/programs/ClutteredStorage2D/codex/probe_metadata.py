"""Probe reset-time metadata and object distributions for ClutteredStorage2DEnv."""

from collections import Counter, defaultdict

from env_client import make_env


def obj_record(state, obj, features):
    return {feature: float(state.get(obj, feature)) for feature in features}


def main():
    env = make_env()
    try:
        print("max_steps", env.max_steps)
        print("action", env.action_space.shape, env.action_space.low,
              env.action_space.high, env.action_space.dtype)
        print("types", [str(t) for t in env.observation_space.types])
        print("type_features", env.observation_space.type_features)
        count_hist = Counter()
        layout_hist = Counter()
        ranges = defaultdict(lambda: [float("inf"), -float("inf")])
        examples = []
        for seed in range(200):
            state, info = env.reset(seed=seed)
            names = state.get_object_names()
            if seed < 5:
                examples.append((seed, names, info))
            if seed in (0, 7):
                print("selected_seed", seed, info)
                for type_name in env.observation_space.types:
                    features = env.observation_space.type_features[type_name]
                    for obj in state.get_objects(type_name):
                        print("selected_object", seed, str(type_name), obj.name,
                              obj_record(state, obj, features))
            for type_name in env.observation_space.types:
                objects = state.get_objects(type_name)
                type_key = str(type_name)
                if "target_block" in type_key:
                    count_hist[len(objects)] += 1
                features = env.observation_space.type_features[type_name]
                for obj in objects:
                    for feature, value in obj_record(state, obj, features).items():
                        key = (type_key, feature)
                        ranges[key][0] = min(ranges[key][0], value)
                        ranges[key][1] = max(ranges[key][1], value)
            shelf_type = env.observation_space.get_type("shelf")
            block_type = env.observation_space.get_type("target_block")
            shelf = next(iter(state.get_objects(shelf_type)))
            blocks = list(state.get_objects(block_type))
            sx = float(state.get(shelf, "x1"))
            sw = float(state.get(shelf, "width1"))
            initial_inside = sum(
                sx <= float(state.get(obj, "x")) <= sx + sw
                and float(state.get(obj, "y")) > 2.5 for obj in blocks)
            layout_hist[(len(blocks), round(sw, 4), initial_inside)] += 1
        print("examples", examples)
        print("block_count_hist", sorted(count_hist.items()))
        print("layout_hist", sorted(layout_hist.items()))
        for object_count in (1, 3, 7):
            state, info = env.reset(seed=0, options={"object_count": object_count})
            shelf = next(iter(state.get_objects(env.observation_space.get_type("shelf"))))
            block_type = env.observation_space.get_type("target_block")
            blocks = list(state.get_objects(block_type))
            shelf_x = float(state.get(shelf, "x1"))
            shelf_w = float(state.get(shelf, "width1"))
            inside = [obj.name for obj in blocks
                      if shelf_x <= float(state.get(obj, "x")) <= shelf_x + shelf_w
                      and float(state.get(obj, "y")) > 2.5]
            print("pinned", object_count, info, "shelf", shelf_x, shelf_w,
                  "inside", inside)
        for key in sorted(ranges, key=lambda item: (item[0], item[1])):
            print("range", key, ranges[key])
    finally:
        env.close()


if __name__ == "__main__":
    main()

import sys
from env_client import make_env

args = list(sys.argv[1:])
count = None
if args[:1] == ["--count"]:
    count = int(args[1]); args = args[2:]
for seed in map(int, args):
    env = make_env()
    options = {} if count is None else {"object_count": count}
    state, _ = env.reset(seed=seed, options=options)
    print("SEED", seed)
    for name in state.get_object_names():
        if name == "robot":
            continue
        obj = state.get_object_from_name(name)
        vals = []
        for feature in ("x", "y", "width", "height"):
            vals.append(round(float(state.get(obj, feature)), 3))
        print(name, vals)
    env.close()

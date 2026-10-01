from env_client import make_env
import sys
env = make_env()
def dump(obs):
    for o in sorted(obs.data, key=lambda o:o.name):
        feats = obs.type_features[o.type]
        print(" ", o.name, {f: round(float(obs.get(o,f)),3) for f in feats if not f.startswith('color')})
for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    obs, info = env.reset(seed=seed)
    print("seed", seed, "info", info)
    dump(obs)
env.close()

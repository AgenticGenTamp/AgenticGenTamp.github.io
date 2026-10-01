from env_client import make_env
import sys
env = make_env()
tf = {t.name: env.observation_space.type_features[t] if hasattr(env.observation_space,'type_features') else None for t in env.observation_space.types}
for seed in [int(x) for x in sys.argv[1:]] or [42]:
    obs,info = env.reset(seed=seed)
    print("=== seed",seed,"info",info)
    for name in sorted(obs.get_object_names()):
        o = obs.get_object_from_name(name)
        feats = tf[o.type.name]
        print(" ",name, o.type.name, {f: round(float(obs.get(o,f)),4) for f in feats})
env.close()

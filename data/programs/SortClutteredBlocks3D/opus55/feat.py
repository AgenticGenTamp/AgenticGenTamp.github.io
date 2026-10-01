from env_client import make_env
env=make_env(); obs,info=env.reset(seed=0)
for n in obs.get_object_names():
    o=obs.get_object_from_name(n)
    if n.startswith("bin") or n.startswith("cube1") or n=="table":
        try: print(n, o.type if hasattr(o,'type') else '', [ (f, round(float(obs.get(o,f)),3)) for f in obs.type_features[o.type]] if hasattr(obs,'type_features') else '')
        except Exception as e: print(n, e)
print(obs.get_object_names())

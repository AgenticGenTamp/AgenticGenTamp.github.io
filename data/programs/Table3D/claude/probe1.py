from env_client import make_env
import numpy as np, json
env = make_env()
obs, info = env.reset(seed=0)
sp = json.load(open("env_spaces.json"))["observation_space"]["types"]
tf = {t["name"]: t["features"] for t in sp}
for name in sorted(obs.get_object_names()):
    o = obs.get_object_from_name(name)
    print(name, o.type.name, {f: round(float(obs.get(o,f)),4) for f in tf[o.type.name]})
print("info", info)
env.close()

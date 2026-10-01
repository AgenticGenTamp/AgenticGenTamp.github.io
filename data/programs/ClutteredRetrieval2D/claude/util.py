import json
from env_client import make_env
TF = {t["name"]: t["features"] for t in json.load(open("env_spaces.json"))["observation_space"]["types"]}
def dump(obs, names=None):
    out={}
    for n in sorted(obs.get_object_names()):
        if names and n not in names: continue
        o=obs.get_object_from_name(n)
        out[n]={f: round(float(obs.get(o,f)),5) for f in TF[o.type.name]}
    return out
def g(obs,name,feat):
    return float(obs.get(obs.get_object_from_name(name),feat))

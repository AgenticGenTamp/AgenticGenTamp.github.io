from env_client import make_env
import json,sys
out={}
for seed in range(1,15):
    env=make_env(); obs,info=env.reset(seed=seed)
    d={}
    for nm in sorted(obs.get_object_names()):
        o=obs.get_object_from_name(nm)
        if nm=="robot":
            d[nm]=[round(obs.get(o,"pos_base_x"),4),round(obs.get(o,"pos_base_y"),4),round(obs.get(o,"pos_base_rot"),4)]
        else:
            d[nm]=[round(obs.get(o,"x"),4),round(obs.get(o,"y"),4),round(obs.get(o,"z"),4)]
    out[seed]=d; env.close()
json.dump(out,open("probe_rew_seeds.json","w"),indent=0)
print(json.dumps(out))

from env_client import make_env
import numpy as np
env=make_env()
tops=set()
for seed in range(200):
    obs,info=env.reset(seed=seed)
    for n in obs.get_object_names():
        if n.startswith("obstacle"):
            o=obs.get_object_from_name(n)
            y=float(obs.get(o,"y")); h=float(obs.get(o,"height")); x=float(obs.get(o,"x")); w=float(obs.get(o,"width"))
            tops.add((round(y,4)==0.0, round(y+h,4), round(x,3), round(w,4)))
bot=[t for t in tops if t[0]]
top=[t for t in tops if not t[0]]
print("num bottom segs",len(bot),"tops of upper segs:", sorted(set(t[1] for t in top))[:5], sorted(set(t[1] for t in top))[-5:])
print("xs:", sorted(set(t[2] for t in tops)), "widths:", sorted(set(t[3] for t in tops)))
env.close()

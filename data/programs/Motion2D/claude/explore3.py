from env_client import make_env
import numpy as np
env = make_env()
mins=[]
for seed in range(60):
    obs,info=env.reset(seed=seed)
    obs_names=[n for n in obs.get_object_names() if n.startswith("obstacle")]
    xs={}
    for n in obs_names:
        o=obs.get_object_from_name(n)
        xs.setdefault(round(float(obs.get(o,"x")),3),[]).append((float(obs.get(o,"y")),float(obs.get(o,"height")),float(obs.get(o,"width"))))
    r=obs.get_object_from_name("robot")
    t=obs.get_object_from_name("target_region")
    gaps=[]
    for x,segs in sorted(xs.items()):
        segs.sort()
        gaps.append((x, round(segs[0][0]+segs[0][1],3), round(segs[1][0],3)))
    print(seed, info['object_count'], "robot",round(float(obs.get(r,"x")),3),round(float(obs.get(r,"y")),3),
          "tgt",round(float(obs.get(t,"x")),3),round(float(obs.get(t,"y")),3),round(float(obs.get(t,"width")),3),round(float(obs.get(t,"height")),3),
          "gaps",gaps)
    for g in gaps: mins.append(g[2]-g[1])
print("min gap", min(mins), "max gap", max(mins))
env.close()

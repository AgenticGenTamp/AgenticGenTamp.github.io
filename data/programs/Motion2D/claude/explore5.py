from env_client import make_env
import numpy as np
env=make_env()
for seed in [11,14,23]:
    obs,info=env.reset(seed=seed)
    r=obs.get_object_from_name("robot"); t=obs.get_object_from_name("target_region")
    tx,ty,tw,th=[float(obs.get(t,f)) for f in ["x","y","width","height"]]
    for cand,name in [((tx,ty),"corner-as-BL->center"),]:
        pass
    gx,gy=tx+tw/2, ty+th/2   # assume bottom-left
    print("seed",seed,"tgt",tx,ty,"goal",gx,gy,"robot",float(obs.get(r,"x")),float(obs.get(r,"y")))
    steps=0; term=False
    for i in range(200):
        x,y=float(obs.get(r,"x")),float(obs.get(r,"y"))
        dx=np.clip(gx-x,-0.05,0.05); dy=np.clip(gy-y,-0.05,0.05)
        obs,rew,term,tr,_=env.step(np.array([dx,dy,0,-0.1,0.0])); steps+=1
        if term: break
        if abs(dx)<1e-6 and abs(dy)<1e-6:
            print("  reached goal pt, no term at",x,y); break
    print("  steps",steps,"term",term,"pos",float(obs.get(r,"x")),float(obs.get(r,"y")))
env.close()

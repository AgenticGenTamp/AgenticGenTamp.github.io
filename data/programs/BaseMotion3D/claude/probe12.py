import numpy as np, json
from env_client import make_env
env=make_env()
T=[]
for s in range(500):
    o,_=env.reset(seed=s); T.append(o[19:22].copy())
T=np.array(T); np.save("targets.npy",T)
print("z uniq", np.unique(T[:,2])[:5], T[:,2].min(), T[:,2].max())
print("x", T[:,0].min(), T[:,0].max(), "y",T[:,1].min(),T[:,1].max())
d=json.load(open("map.json")); free=set(map(tuple,d["free"]))
F=np.array(sorted(free))*0.1
bad=0
for t in T:
    dist=np.min(np.hypot(F[:,0]-t[0],F[:,1]-t[1]))
    if dist>0.05: bad+=1; print("far",t.round(3),round(float(dist),3))
print("bad",bad,"/",len(T))

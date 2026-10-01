from env_client import make_env
import numpy as np
env = make_env()
def d(obs,n):
    o=obs.get_object_from_name(n); return {f: float(obs.get(o,f)) for f in obs.type_features[o.type]}
mx=0;mn=9;mxo=0;mno=9;mnb=9;mxb=0
for seed in range(200):
    obs,info=env.reset(seed=seed)
    ts=d(obs,'target_surface'); tb=d(obs,'target_block')
    mx=max(mx,ts['x']+ts['width']); mn=min(mn,ts['x'])
    mxb=max(mxb,tb['x']+tb['width']); mnb=min(mnb,tb['x'])
    for i in range(info['object_count']):
        ob=d(obs,'obstruction%d'%i); mxo=max(mxo,ob['x']+ob['width']); mno=min(mno,ob['x'])
print("surface x range",mn,mx)
print("block x range",mnb,mxb)
print("obst x range",mno,mxo)
env.close()

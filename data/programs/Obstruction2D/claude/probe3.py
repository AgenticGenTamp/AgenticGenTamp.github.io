from env_client import make_env
import numpy as np
env = make_env()
def d(obs,n):
    o=obs.get_object_from_name(n); return {f: round(float(obs.get(o,f)),4) for f in obs.type_features[o.type]}
obs,info=env.reset(seed=42)
print("blk",d(obs,'target_block'))
# retract arm fully
for i in range(5): obs,*_=env.step(np.array([0,0,0,-0.1,0]))
print("retracted", d(obs,'robot'))
# move to above block center x=1.171
tb=d(obs,'target_block'); cx=tb['x']+tb['width']/2
r=d(obs,'robot')
for i in range(40):
    r=d(obs,'robot')
    dx=np.clip(cx-r['x'],-0.05,0.05)
    if abs(dx)<1e-4: break
    obs,*_=env.step(np.array([dx,0,0,0,0]))
print("above", d(obs,'robot'))
# move down until blocked
prev=None
for i in range(40):
    obs,*_=env.step(np.array([0,-0.05,0,0,0]))
    r=d(obs,'robot')
    if prev is not None and abs(r['y']-prev)<1e-6:
        print("blocked at y",r['y'],"iter",i); break
    prev=r['y']
print("robot",d(obs,'robot'))
# extend arm
for i in range(10):
    obs,*_=env.step(np.array([0,0,0,0.02,0]))
    print("arm",d(obs,'robot')['arm_joint'], d(obs,'robot')['y'])
env.close()

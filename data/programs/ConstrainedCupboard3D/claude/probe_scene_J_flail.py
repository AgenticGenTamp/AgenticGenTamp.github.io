import numpy as np
from env_client import make_env
def feats(obs,name):
    o=obs.get_object_from_name(name); return dict(zip(obs.type_features[o.type],[float(v) for v in obs.data[o]]))
rng=np.random.default_rng(1)
env=make_env(); obs,info=env.reset(seed=0,options={'object_count':1})
c0=feats(obs,'cuboid_0'); start=np.array([c0['x'],c0['y'],c0['z'],c0['qw'],c0['qz']])
print("rod0",[round(v,4) for v in start])
best=0.0; rews=set()
for i in range(3000):
    rb=feats(obs,'robot'); a=np.zeros(11,dtype=np.float32)
    # keep base wandering within 0.6m of rod
    a[0]=np.clip(1.0*(c0['x']-0.4+rng.uniform(-0.25,0.25)-rb['pos_base_x']),-0.1,0.1)
    a[1]=np.clip(1.0*(c0['y']+rng.uniform(-0.25,0.25)-rb['pos_base_y']),-0.1,0.1)
    a[3:10]=rng.uniform(-0.1,0.1,7)
    a[10]=float(rng.random()<0.5)
    obs,r,term,trunc,info=env.step(a); rews.add(round(r,6))
    c=feats(obs,'cuboid_0')
    d=abs(c['x']-c0['x'])+abs(c['y']-c0['y'])+abs(c['z']-c0['z'])
    if d>best+0.002:
        best=d; print(f" step {i} rod moved d={d:.4f} pos=({c['x']:.4f},{c['y']:.4f},{c['z']:.4f}) r={r}")
    if term or trunc: print("END",i,term,trunc,r); break
print("rewards seen",sorted(rews),"max rod displacement",best)
print("final joints",[round(feats(obs,'robot')[f'pos_arm_joint{k}'],3) for k in range(1,8)],"grip",round(feats(obs,'robot')['pos_gripper'],3))
env.close()

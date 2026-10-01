import numpy as np, itertools, json
from env_client import make_env

def feats(obs, name):
    o = obs.get_object_from_name(name)
    return dict(zip(obs.type_features[o.type], [float(v) for v in obs.data[o]]))

def go(env, obs, base=None, arm=None, grip=None, steps=60, log=None):
    for i in range(steps):
        rb = feats(obs,'robot'); a=np.zeros(11,dtype=np.float32)
        err=0.0
        if base is not None:
            for k,(f,idx) in enumerate([('pos_base_x',0),('pos_base_y',1),('pos_base_rot',2)]):
                e=base[k]-rb[f]; err=max(err,abs(e)); a[idx]=np.clip(2*e,-0.1,0.1)
        if arm is not None:
            for j in range(7):
                if arm[j] is None: continue
                e=arm[j]-rb[f'pos_arm_joint{j+1}']; err=max(err,abs(e)); a[3+j]=np.clip(2*e,-0.1,0.1)
        if grip is not None: a[10]=grip
        obs,r,term,trunc,info=env.step(a)
        if log is not None: log.append((r, feats(obs,'cuboid_0')))
        if err<0.01: break
    return obs, r

CONFIGS = {
 "A": [0,0.5,3.14,-2.0,0,-0.8,1.57],
 "B": [0,1.0,3.14,-1.5,0,-0.5,1.57],
 "C": [0,0.6,3.14,-2.6,0,0.5,1.57],
 "D": [0,1.4,3.14,-1.0,0,-1.0,1.57],
 "E": [0,0.0,3.14,-1.6,0,1.6,1.57],
}
env=make_env(); obs,info=env.reset(seed=0, options={'object_count':1})
c0=feats(obs,'cuboid_0'); print("rod at", round(c0['x'],3), round(c0['y'],3))
paths=[]
for name,cfg in CONFIGS.items():
    obs,info=env.reset(seed=0, options={'object_count':1})
    c0=feats(obs,'cuboid_0')
    obs,r=go(env,obs,base=[c0['x']-0.45, c0['y'], 0.0],steps=200)
    obs,r=go(env,obs,arm=cfg,grip=1.0,steps=120)
    c=feats(obs,'cuboid_0')
    moved=abs(c['x']-c0['x'])+abs(c['y']-c0['y'])+abs(c['z']-c0['z'])
    # creep base forward
    log=[]
    rmin=r
    for k in range(30):
        rb=feats(obs,'robot'); obs,r=go(env,obs,base=[rb['pos_base_x']+0.03, c0['y'],0.0],arm=cfg,grip=1.0,steps=3,log=log)
    c2=feats(obs,'cuboid_0')
    rs=set(round(x[0],6) for x in log)
    print(f"{name} cfg={cfg} moved_on_pose={moved:.4f} rod_after_creep=({c2['x']:.4f},{c2['y']:.4f},{c2['z']:.4f}) rewards={sorted(rs)}")
    paths.append(env.render_state(state=obs,label=f"H_{name}"))
print("PATHS",paths)
env.close()

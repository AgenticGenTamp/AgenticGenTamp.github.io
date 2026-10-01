import sys, numpy as np
from env_client import make_env

def feats(obs, name):
    o = obs.get_object_from_name(name)
    return dict(zip(obs.type_features[o.type], [float(v) for v in obs.data[o]]))

env = make_env()
obs, info = env.reset(seed=0, options={'object_count':1})
c = feats(obs,'cuboid_0'); print("cuboid", round(c['x'],3), round(c['y'],3), round(c['z'],3))
paths=[]
paths.append(env.render_state(state=obs, label="G_reset"))
# drive base to just behind cuboid
tx, ty = c['x']-0.30, c['y']
for i in range(200):
    rb=feats(obs,'robot'); a=np.zeros(11,dtype=np.float32)
    a[0]=np.clip(2*(tx-rb['pos_base_x']),-0.1,0.1); a[1]=np.clip(2*(ty-rb['pos_base_y']),-0.1,0.1)
    a[2]=np.clip(-2*rb['pos_base_rot'],-0.1,0.1)
    obs,r,term,trunc,info=env.step(a)
    if abs(tx-rb['pos_base_x'])<0.005 and abs(ty-rb['pos_base_y'])<0.005: break
print("after base move: base", {k:round(v,3) for k,v in feats(obs,'robot').items() if k.startswith('pos_base')}, "cub", {k:round(v,4) for k,v in feats(obs,'cuboid_0').items() if k in 'xyz'}, "r",r)
paths.append(env.render_state(state=obs, label="G_base_near"))
# lower the arm: sweep joint2 positive (shoulder down), joint4 toward 0
for j, delta, n in [(4, +0.1, 20), (6, +0.1, 20)]:
    for i in range(n):
        a=np.zeros(11,dtype=np.float32); a[j]=delta
        obs,r,term,trunc,info=env.step(a)
    print(f"after joint idx{j}: joints", [round(feats(obs,'robot')[f'pos_arm_joint{k}'],3) for k in range(1,8)], "cub", {k:round(v,4) for k,v in feats(obs,'cuboid_0').items() if k in 'xyz'}, "r", r)
    paths.append(env.render_state(state=obs, label=f"G_j{j}"))
# now push base forward with arm lowered
for i in range(60):
    a=np.zeros(11,dtype=np.float32); a[0]=0.1
    obs,r,term,trunc,info=env.step(a)
    if abs(r+1.0)>1e-9: print("RDIFF",i,r)
print("after push: base", {k:round(v,3) for k,v in feats(obs,'robot').items() if k.startswith('pos_base')}, "cub", {k:round(v,4) for k,v in feats(obs,'cuboid_0').items() if k in 'xyz'}, "r", r)
paths.append(env.render_state(state=obs, label="G_after_push"))
print("PATHS", paths)
env.close()

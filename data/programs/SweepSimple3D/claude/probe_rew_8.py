import sys
from env_client import make_env
import numpy as np
mode=sys.argv[1]; seed=int(sys.argv[2])
rng=np.random.default_rng(seed+7)
env=make_env(); os_=env.observation_space; T=os_.get_type
obs,info=env.reset(seed=seed)
def snap(obs):
    d={}
    for o in obs.get_objects(T("mujoco_movable_object")):
        d[o.name]=np.array([float(obs.get(o,'x')),float(obs.get(o,'y')),float(obs.get(o,'z'))])
    r=obs.get_objects(T("mujoco_tidybot_robot"))[0]
    d['B']=np.array([float(obs.get(r,'pos_base_x')),float(obs.get(r,'pos_base_y')),0.0])
    return d
s0=snap(obs); print("t0",{k:np.round(v,3).tolist() for k,v in s0.items()},flush=True)
def step(a,tag,i):
    global obs
    obs,r,te,tr,inf=env.step(a); s=snap(obs)
    dmax=max(np.linalg.norm(s[k][:2]-s0[k][:2]) for k in s if k.startswith('cube'))
    if float(r)!=-1.0 or i%10==0 or dmax>0.05:
        print(tag,i,repr(float(r)),"dmax",round(dmax,3),{k:np.round(v,3)[:2].tolist() for k,v in s.items()},te,tr,flush=True)
    return te or tr
# drive base near wiper (0.35 behind in y, aligned x)
tx,ty=s0['wiper_0'][0],s0['wiper_0'][1]+0.35
for i in range(20):
    b=snap(obs)['B']; d=np.clip([tx-b[0],ty-b[1]],-0.08,0.08)
    if max(abs(d))<0.005: break
    a=np.zeros(11,dtype=np.float32); a[0],a[1]=d
    obs,r,te,tr,inf=env.step(a)
print("staged",np.round(snap(obs)['B'],3).tolist(),flush=True)
if mode=="flail":
    for i in range(120):
        a=np.zeros(11,dtype=np.float32); a[3:10]=rng.uniform(-0.1,0.1,7); a[10]=1.0
        a[0:2]=rng.uniform(-0.04,0.04,2)
        if step(a,"flail",i): break
elif mode=="rot":
    for k,sg in enumerate([1,-1,1,-1]):
        for i in range(25):
            a=np.zeros(11,dtype=np.float32); a[2]=sg*0.1; a[10]=1.0
            if step(a,f"rot{k}",i): break
env.close()

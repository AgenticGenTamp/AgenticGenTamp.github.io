import sys
from env_client import make_env
import numpy as np
mode=sys.argv[1]; seed=int(sys.argv[2])
env=make_env(); os_=env.observation_space; T=os_.get_type
obs,info=env.reset(seed=seed)
def snap(obs):
    d={}
    for o in obs.get_objects(T("mujoco_movable_object")):
        d[o.name]=(round(float(obs.get(o,'x')),3),round(float(obs.get(o,'y')),3),round(float(obs.get(o,'z')),3))
    r=obs.get_objects(T("mujoco_tidybot_robot"))[0]
    d['B']=(round(float(obs.get(r,'pos_base_x')),3),round(float(obs.get(r,'pos_base_y')),3))
    return d
s0=snap(obs); print("t0",s0,flush=True)
grip=1.0 if mode=="grasp" else 0.0
def go(dx,dy,n,g=0.0,tag=""):
    global obs
    a=np.zeros(11,dtype=np.float32); a[0]=dx; a[1]=dy; a[10]=g
    for i in range(n):
        obs,r,te,tr,inf=env.step(a)
        print(tag,i,repr(float(r)),snap(obs),te,tr,flush=True)
        if te or tr: return True
    return False
# align x with wiper
tgt=s0['wiper_0'][0]
for i in range(12):
    dx=float(np.clip(tgt-snap(obs)['B'][0],-0.1,0.1))
    if abs(dx)<0.002: break
    obs,r,te,tr,inf=env.step(np.array([dx]+[0]*10,dtype=np.float32))
if mode=="slow":
    go(0,-0.03,60,0.0,"slow")
elif mode=="grasp":
    go(0,-0.1,6,0.0,"appr")   # base y ~1.26
    go(0,-0.03,8,0.0,"appr2")
    go(0,0,3,1.0,"close")
    go(0,-0.03,30,1.0,"push")
env.close()

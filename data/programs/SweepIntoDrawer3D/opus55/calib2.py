import numpy as np, sys
from env_client import make_env
from kin import *
np.set_printoptions(precision=4, suppress=True, linewidth=150)
env = make_env()
obs, info = env.reset(seed=0)
base = obs[125:128].copy(); q = obs[128:135].copy()
def goto(qd, n=60, grip=0.0):
    global obs
    for t in range(n):
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-obs[128:135],-0.1,0.1); a[10]=grip
        obs,*_=env.step(a)
        if np.max(np.abs(obs[128:135]-qd))<2e-3: break
    return obs[128:135]
ci=int(sys.argv[1]) if len(sys.argv)>1 else 0
c=obs[16*ci:16*ci+3]; qw,qz=obs[16*ci+3],obs[16*ci+6]
yaw=2*np.arctan2(qz,qw); yaw=(yaw+np.pi/4)%(np.pi/2)-np.pi/4
qc=q.copy()
for z,g in [(0.56,0),(0.50,0),(0.46,0),(0.435,0),(0.435,1),(0.55,1)]:
    qd,err=ik(base,qc,np.array([c[0],c[1],z]),R_down(yaw))
    qa=goto(qd,n=60,grip=g); qc=qd
    for _ in range(5 if g else 0):
        obs,*_=env.step(np.concatenate([np.clip(qd-obs[128:135],-.1,.1)*0,[0]*0]).tolist()+[0]*0 if False else np.r_[np.zeros(3),np.clip(qd-obs[128:135],-.1,.1),g].astype(np.float32))
    pa=fk_world(base,obs[128:135])[:3,3]
    print(f"z={z} g={g} tool={pa} cube={obs[16*ci:16*ci+3]} grip={obs[135]:.3f}")
T=fk_world(base,obs[128:135]); print("cube in tool frame:", T[:3,:3].T@(obs[16*ci:16*ci+3]-T[:3,3]))
np.save('calib_obs.npy',obs)
env.close()

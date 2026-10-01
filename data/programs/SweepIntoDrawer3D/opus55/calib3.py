import numpy as np, sys
from env_client import make_env
import kin
from kin import *
np.set_printoptions(precision=4, suppress=True, linewidth=150)
env = make_env()
obs, info = env.reset(seed=0)
base = obs[125:128].copy(); q = obs[128:135].copy()
def goto(qd, n=80, grip=0.0):
    global obs
    for t in range(n):
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-obs[128:135],-0.1,0.1); a[10]=grip
        obs,*_=env.step(a)
        if np.max(np.abs(obs[128:135]-qd))<1e-3: break
    for t in range(5):
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-obs[128:135],-0.1,0.1); a[10]=grip
        obs,*_=env.step(a)
    return obs[128:135]
ci=0
c=obs[16*ci:16*ci+3]; qw,qz=obs[16*ci+3],obs[16*ci+6]
yaw=2*np.arctan2(qz,qw); yaw=(yaw+np.pi/4)%(np.pi/2)-np.pi/4
qc=q.copy()
for z,g in [(0.56,0),(0.46,0),(0.44,0),(0.44,1),(0.55,1)]:
    qd,err=ik(base,qc,np.array([c[0],c[1],z]),R_down(yaw)); qa=goto(qd,grip=g); qc=qd
data=[]
rng=np.random.default_rng(0)
for k in range(12):
    p=np.array([0.8+rng.uniform(-.1,.1), -0.07+rng.uniform(-.15,.15), 0.6+rng.uniform(-.05,.15)])
    R=R_down(yaw+rng.uniform(-1,1))
    # tilt
    ax=rng.normal(size=3); ax/=np.linalg.norm(ax); ang=rng.uniform(0,0.5)
    K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    R=(np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*K@K)@R
    qd,err=ik(base,qc,p,R); qa=goto(qd,grip=1); qc=qd
    data.append((obs[125:128].copy(), qa.copy(), obs[16*ci:16*ci+3].copy()))
    T=fk_world(obs[125:128],qa); print(k, err, T[:3,:3].T@(obs[16*ci:16*ci+3]-T[:3,3]), obs[16*ci+2])
np.save('calib_data.npy', np.array([np.concatenate(d) for d in data]))
env.close()

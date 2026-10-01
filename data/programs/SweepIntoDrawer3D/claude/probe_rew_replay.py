import numpy as np, sys
from env_client import make_env
np.set_printoptions(precision=3, suppress=True, linewidth=250)
seed=3; rng=np.random.default_rng(500+seed)
env=make_env(); obs,_=env.reset(seed=seed)
lo=np.array([-3.0,-2.4,-3.0,-2.6,-3.0,-2.2,-3.0]); hi=np.array([3.0,2.4,3.0,2.6,3.0,2.2,3.0])
qt=obs[128:135].copy(); bt=obs[125:128].copy(); g=0.0
hit=None
for k in range(2000):
    if k%25==0:
        qt=rng.uniform(lo,hi); bt=np.array([rng.uniform(1.15,1.30),rng.uniform(-0.9,0.9),rng.uniform(2.9,3.4)]); g=float(rng.random()<0.5)
    a=np.zeros(11)
    a[:3]=np.clip(bt-obs[125:128],-0.1,0.1)
    a[3:10]=np.clip(qt-obs[128:135],-0.1,0.1); a[10]=g
    obs,r,t,tr,i=env.step(a)
    if obs[103]>0.55: hit=k; break
print("opened at step",hit,"drawer",np.round(obs[103:109],3),"base",np.round(obs[125:128],3),"r",r)
print(env.render_state(state=obs.tolist(),label="opened_real"))
# retract base and scan
def goto(env,obs,bx,by,steps=60,qhome=np.array([0,-0.349,3.142,-2.548,0,-0.873,1.571])):
    for k in range(steps):
        a=np.zeros(11); a[0]=np.clip(bx-obs[125],-0.1,0.1); a[1]=np.clip(by-obs[126],-0.1,0.1)
        a[2]=np.clip(3.1-obs[127],-0.1,0.1); a[3:10]=np.clip(qhome-obs[128:135],-0.1,0.1); a[10]=0.0
        obs,r,t,tr,_=env.step(a)
    return obs,r
obs,r=goto(env,obs,1.9,obs[126]); print("retracted drawer",np.round(obs[103:109],3),"base",np.round(obs[125:128],3))
for ytar in [-0.9,-0.6,-0.3,0.0,0.3,0.6,0.9]:
    obs,r=goto(env,obs,1.9,ytar,60)
    for k in range(60):
        a=np.zeros(11); a[0]=-0.1; a[1]=np.clip(ytar-obs[126],-0.1,0.1)
        obs,r,t,tr,_=env.step(a)
    print("y",ytar,"blocked_x",round(float(obs[125]),3),"drawer",np.round(obs[103:109],3),"r",r)
env.close()

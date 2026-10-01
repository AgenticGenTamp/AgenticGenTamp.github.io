import numpy as np, sys
from env_client import make_env
np.set_printoptions(precision=3, suppress=True, linewidth=250)
seed=int(sys.argv[1]); rng=np.random.default_rng(500+seed)
env=make_env(); obs,_=env.reset(seed=seed)
lo=np.array([-3.0,-2.4,-3.0,-2.6,-3.0,-2.2,-3.0]); hi=np.array([3.0,2.4,3.0,2.6,3.0,2.2,3.0])
seen={}
qt=obs[128:135].copy(); bt=obs[125:128].copy()
prev=obs[103:109].copy()
for k in range(2000):
    if k%25==0:
        qt=rng.uniform(lo,hi); bt=np.array([rng.uniform(1.15,1.30),rng.uniform(-0.9,0.9),rng.uniform(2.9,3.4)]); g=float(rng.random()<0.5)
    a=np.zeros(11)
    a[0]=np.clip(bt[0]-obs[125],-0.1,0.1); a[1]=np.clip(bt[1]-obs[126],-0.1,0.1); a[2]=np.clip(bt[2]-obs[127],-0.1,0.1)
    a[3:10]=np.clip(qt-obs[128:135],-0.1,0.1); a[10]=g
    obs,r,t,tr,i=env.step(a)
    d=obs[103:109]
    for j in range(6):
        if d[j]>0.05 and d[j]>seen.get(j,(0,))[0]:
            seen[j]=(round(float(d[j]),3),round(float(obs[126]),2),round(float(r),4))
    if abs(r+1)>1e-6: print("REWARD CHANGE",k,r,np.round(obs[:80].reshape(5,16)[:,:3],3).tolist())
print("seed",seed,"drawer_idx -> (maxopen, base_y_at_that_time, reward):",seen)
print("final drawer",np.round(obs[103:109],3),"maxcubemove ok")
env.close()

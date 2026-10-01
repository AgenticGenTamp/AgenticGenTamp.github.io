import numpy as np, sys
from env_client import make_env
np.set_printoptions(precision=3, suppress=True, linewidth=250)
seed=int(sys.argv[1]); rng=np.random.default_rng(100+seed)
env=make_env(); obs,_=env.reset(seed=seed)
c0=obs[:80].reshape(5,16)[:,:3].copy(); w0=obs[147:150].copy()
lo=np.array([-3.0,-2.4,-3.0,-2.6,-3.0,-2.2,-3.0]); hi=np.array([3.0,2.4,3.0,2.6,3.0,2.2,3.0])
best=0; events=[]
qt=obs[128:135].copy(); bt=obs[125:128].copy()
for k in range(1500):
    if k%25==0:
        qt=rng.uniform(lo,hi); bt=np.array([rng.uniform(1.15,1.35),rng.uniform(-0.6,0.3),rng.uniform(2.8,3.5)]); g=float(rng.random()<0.5)
    a=np.zeros(11)
    a[0]=np.clip(bt[0]-obs[125],-0.1,0.1); a[1]=np.clip(bt[1]-obs[126],-0.1,0.1); a[2]=np.clip(bt[2]-obs[127],-0.1,0.1)
    a[3:10]=np.clip(qt-obs[128:135],-0.1,0.1); a[10]=g
    obs,r,t,tr,i=env.step(a)
    c=obs[:80].reshape(5,16)[:,:3]; d=np.linalg.norm(c-c0,axis=1)
    dw=np.linalg.norm(obs[147:150]-w0); dr=np.abs(obs[103:109]).max()
    if d.max()>best+0.01 or abs(r+1)>1e-6 or dw>0.02 or dr>0.03:
        best=max(best,d.max())
        events.append((k,round(r,4),np.round(d,3).tolist(),round(float(dw),3),round(float(dr),3),np.round(obs[125:128],2).tolist()))
    if t or tr: print("END",k,t,tr); break
print("seed",seed,"maxcubemove",round(best,3),"nev",len(events))
for e in events[:15]: print("  ",e)
print("final cubes",np.round(obs[:80].reshape(5,16)[:,:3],3).tolist())
print("final wiper",np.round(obs[147:150],3),"drawer",np.round(obs[103:109],3))
env.close()

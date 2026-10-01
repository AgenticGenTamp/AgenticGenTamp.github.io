import numpy as np, threading, json
from env_client import make_env
from ctrl2 import moveto, RFWD
np.set_printoptions(precision=3,suppress=True,linewidth=200)
out={}
def job(ztip):
    env=make_env(); obs,_=env.reset(seed=0)
    c3y=0.074  # local y of cube3
    res=[]
    obs,d=moveto(env,obs,[0.30,c3y,ztip],R=RFWD,steps=300,grip=0.0)
    res.append(('approach',d))
    prev=obs[:80].reshape(5,16)[:,:3].copy()
    for x in np.arange(0.32,0.70,0.02):
        obs,d=moveto(env,obs,[x,c3y,ztip],R=RFWD,steps=40,grip=0.0)
        c=obs[:80].reshape(5,16)[:,:3]
        mv=np.linalg.norm(c-prev,axis=1)
        res.append((round(x,2),d['err'],d['ikres'],d['clip'],[round(v,3) for v in mv]))
        prev=c.copy()
    env.close(); out[ztip]=res
ths=[threading.Thread(target=job,args=(z,)) for z in [0.50,0.47,0.44,0.41]]
[t.start() for t in ths]; [t.join() for t in ths]
for z in sorted(out): 
    print("=== ztip",z)
    for r in out[z]:
        if r[0]=='approach': print("  approach",r[1])
        elif max(r[4])>0.004 or r[1]>0.05 or r[3]>0.01: print("  ",r)

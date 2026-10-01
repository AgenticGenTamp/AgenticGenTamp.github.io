import numpy as np, threading
from env_client import make_env
from ctrl2 import moveto, RFWD
np.set_printoptions(precision=3,suppress=True,linewidth=200)
out={}
def job(z):
    env=make_env(); obs,_=env.reset(seed=0)
    bx=obs[125]
    rows=[]
    for y in [-0.30,-0.15,0.0,0.15,0.30]:
        obs,d=moveto(env,obs,[0.22,y,z],R=RFWD,steps=250,grip=0.0)
        if d['clip']>0.02 or d['err']>0.05:
            rows.append((y,'noreach',round(d['err'],3))); continue
        stall=None
        for lx in np.arange(0.24,0.60,0.02):
            obs,d=moveto(env,obs,[lx,y,z],R=RFWD,steps=45,grip=0.0)
            if d['clip']>0.02: stall=('clip',round(lx,2)); break
            if d['err']>0.03: stall=('hit',round(lx,2),round(bx-lx,3)); break
        rows.append((y,stall,np.round(obs[103:109],3)))
    env.close(); out[z]=rows
ths=[threading.Thread(target=job,args=(z,)) for z in [0.42,0.36,0.30,0.25]]
[t.start() for t in ths]; [t.join() for t in ths]
for z in sorted(out):
    print("=== z",z)
    for r in out[z]: print("   ",r)

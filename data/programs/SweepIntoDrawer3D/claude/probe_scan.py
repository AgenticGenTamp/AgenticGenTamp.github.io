import numpy as np, threading
from env_client import make_env
from ctrl2 import moveto, RFWD, RDOWN
np.set_printoptions(precision=3,suppress=True,linewidth=200)
out={}
def job(z):
    env=make_env(); obs,_=env.reset(seed=0)
    bx=obs[125]
    rows=[]
    for y in [-0.30,-0.15,0.0,0.15,0.30]:
        # retract
        obs,d=moveto(env,obs,[bx-1.20,y,z],R=RFWD,steps=200,grip=0.0)
        if d['clip']>0.02 or d['err']>0.05:
            rows.append((y,'noreach',d)); continue
        stall=None
        for wx in np.arange(1.18,0.86,-0.02):
            obs,d=moveto(env,obs,[bx-wx,y,z],R=RFWD,steps=45,grip=0.0)
            if d['clip']>0.02: stall=('clip',round(wx,2)); break
            if d['err']>0.03: stall=('hit',round(wx,2)); break
        dr=obs[103:109].copy()
        rows.append((y,stall,np.round(dr,3)))
    env.close(); out[z]=rows
ths=[threading.Thread(target=job,args=(z,)) for z in [0.42,0.36,0.30,0.25]]
[t.start() for t in ths]; [t.join() for t in ths]
for z in sorted(out):
    print("=== z",z)
    for r in out[z]: print("   ",r)

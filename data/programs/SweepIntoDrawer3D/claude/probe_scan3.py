import numpy as np, threading
from env_client import make_env
from ctrl2 import moveto, move_base, RFWD
np.set_printoptions(precision=3,suppress=True,linewidth=200)
out={}
BX=1.75
def job(z):
    env=make_env(); obs,_=env.reset(seed=0)
    obs=move_base(env,obs,[BX,obs[126],obs[127]],steps=40,grip=0.0)
    bx,by,yaw=obs[125:128]
    rows=[]
    for y in [-0.30,-0.15,0.0,0.15,0.30]:
        obs,d=moveto(env,obs,[0.52,y,z],R=RFWD,steps=250,grip=0.0)
        if d['clip']>0.02 or d['err']>0.05:
            rows.append((y,'noreach',round(d['err'],3),round(d['clip'],3))); continue
        stall=None
        for lx in np.arange(0.54,0.95,0.02):
            obs,d=moveto(env,obs,[lx,y,z],R=RFWD,steps=45,grip=0.0)
            if d['clip']>0.02: stall=('clip',round(lx,2)); break
            if d['err']>0.03: stall=('hit',round(lx,2),round(bx-lx,3)); break
        rows.append((y,stall,np.round(obs[103:109],3)))
    env.close(); out[z]=(rows,[bx,by,yaw])
ths=[threading.Thread(target=job,args=(z,)) for z in [0.42,0.36,0.30,0.24]]
[t.start() for t in ths]; [t.join() for t in ths]
for z in sorted(out):
    print("=== z",z,"base",np.round(out[z][1],3))
    for r in out[z][0]: print("   ",r)

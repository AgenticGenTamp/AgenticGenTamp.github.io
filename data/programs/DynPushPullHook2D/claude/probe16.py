from env_client import make_env
from ctl import act, rget, oget
import numpy as np
env=make_env()
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def trial(seed,d,lat,part='bar'):
    obs,info=env.reset(seed=seed)
    def step(a):
        nonlocal obs
        obs,_,_,_,_=env.step(a)
    hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
    if part=='bar':
        ang=hth; L=1.4; off=-0.0533
    else:
        ang=hth+np.pi/2; L=0.583; off=-0.0533  # stub: approach from below, arm points +y local
    u=np.array([np.cos(ang),np.sin(ang)]); n=np.array([-np.sin(ang),np.cos(ang)])
    O=np.array([hx,hy])
    if part=='bar':
        fe=O-1.4*np.array([np.cos(hth),np.sin(hth)])+(-0.0533+lat)*np.array([-np.sin(hth),np.cos(hth)])
    else:
        # stub free end (local (-0.0533, -0.583)) ; approach direction local +y => u=hth+90deg
        lo=np.array([np.cos(hth),np.sin(hth)]); ln=np.array([-np.sin(hth),np.cos(hth)])
        fe=O+(-0.0533+lat)*lo-0.583*ln
    base=fe-d*u
    for _ in range(60): step(act(dth=wrap(ang-rget(obs,'theta')),dg=0.02,da=-0.1))
    # travel: first to a staging point away along -u at safe y
    stag=base-0.0*u
    for _ in range(500):
        dx=np.clip(stag[0]-rget(obs,'x'),-0.05,0.05); dy=np.clip(0.26-rget(obs,'y'),-0.05,0.05)
        if abs(dx)<0.003 and abs(dy)<0.003: break
        step(act(dx=dx,dy=dy))
    for _ in range(500):
        dy=np.clip(base[1]-rget(obs,'y'),-0.05,0.05)
        if abs(dy)<0.003: break
        step(act(dy=dy))
    h0=np.array([oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')])
    for _ in range(30): step(act(da=0.1))
    moved1=np.abs(np.array([oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')])-h0).max()
    res=None
    for i in range(15):
        step(act(dg=-0.02))
        if oget(obs,'hook','held')>0.5:
            res=round(rget(obs,'finger_gap'),3); break
    mv=np.abs(np.array([oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')])-h0).max()
    return res, round(moved1,4), round(mv,4)
for part in ['bar','stub']:
  for d in [0.50,0.55,0.60,0.65,0.69]:
    for lat in [0.0,0.03,-0.03]:
        r=trial(42,d,lat,part)
        print(part,d,lat,"held@",r[0],"movedOnExtend",r[1],"movedTotal",r[2])
env.close()

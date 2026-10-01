import numpy as np
from env_client import make_env
from ctl import act, rget, oget
from grasp import wrap
env=make_env()
obs,info=env.reset(seed=42)
def step(a):
    global obs
    obs,_,_,_,_=env.step(a)
hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
fe0=np.array([hx,hy])-1.4*u-0.0533*nv
stand=np.array([hx,hy])+(-0.107-0.275)*u-0.45*nv
for _ in range(60): step(act(dth=wrap(np.pi+hth-rget(obs,'theta')),da=-0.1,dg=-0.02))
tgt=stand-0.3*u
for _ in range(300):
    dx=np.clip(tgt[0]-rget(obs,'x'),-0.05,0.05); dy=np.clip(tgt[1]-rget(obs,'y'),-0.05,0.05)
    if abs(dx)<0.004 and abs(dy)<0.004: break
    step(act(dx=dx,dy=dy))
for _ in range(12): step(act(dx=0.05))
print("hook now",round(oget(obs,'hook','x'),3),round(oget(obs,'hook','y'),3),round(oget(obs,'hook','theta'),3),"orig fe",np.round(fe0,3))
for _ in range(10): step(act(dx=-0.05))
# grasp at OLD fe location
d=0.31
base=fe0-d*u
for _ in range(80): step(act(dth=wrap(hth-rget(obs,'theta')),da=-0.1,dg=0.02))
for _ in range(300):
    dx=np.clip(base[0]-0.5-rget(obs,'x'),-0.05,0.05); dy=np.clip(0.3-rget(obs,'y'),-0.05,0.05)
    if abs(dx)<0.004 and abs(dy)<0.004: break
    step(act(dx=dx,dy=dy))
for _ in range(300):
    dy=np.clip(base[1]-rget(obs,'y'),-0.05,0.05)
    if abs(dy)<0.004: break
    step(act(dy=dy))
for _ in range(300):
    dx=np.clip(base[0]-rget(obs,'x'),-0.05,0.05)
    if abs(dx)<0.002: break
    step(act(dx=dx))
print("robot at",round(rget(obs,'x'),3),round(rget(obs,'y'),3),"target",np.round(base,3))
for k in range(12):
    step(act(dg=-0.02))
    if oget(obs,'hook','held')>0.5:
        print("HELD AT OLD LOCATION at gap",round(rget(obs,'finger_gap'),3)); break
else: print("no grasp at old location")
env.close()

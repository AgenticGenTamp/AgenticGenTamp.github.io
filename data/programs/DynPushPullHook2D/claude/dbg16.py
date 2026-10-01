import numpy as np
from env_client import make_env
from ctl import act, rget, oget
from grasp import wrap
env=make_env()
obs,info=env.reset(seed=42)
def step(a):
    global obs
    obs,_,_,_,_=env.step(a)
def H(): return np.array([oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')])
hx,hy,hth=H()
u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
stand=np.array([hx,hy])+(-0.107-0.275)*u-0.45*nv
for _ in range(60): step(act(dth=wrap(np.pi+hth-rget(obs,'theta')),da=-0.1,dg=-0.02))
tgt=stand-0.3*u
for _ in range(300):
    dx=np.clip(tgt[0]-rget(obs,'x'),-0.05,0.05); dy=np.clip(tgt[1]-rget(obs,'y'),-0.05,0.05)
    if abs(dx)<0.004 and abs(dy)<0.004: break
    step(act(dx=dx,dy=dy))
for _ in range(12): step(act(dx=0.05))
for _ in range(10): step(act(dx=-0.05))
for _ in range(40): step(act())
hx,hy,hth=H(); print("hook settled",round(hx,4),round(hy,4),round(hth,4))
u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
for tx in [1.9,2.3,2.7]:
    for _ in range(80):
        d=wrap(np.pi/2-rget(obs,'theta'))
        if abs(d)<0.01: break
        step(act(dth=d,da=-0.1,dg=-0.02))
    for _ in range(400):
        dx=np.clip(tx-rget(obs,'x'),-0.05,0.05); dy=np.clip(1.45-rget(obs,'y'),-0.05,0.05)
        if abs(dx)<0.003 and abs(dy)<0.003: break
        step(act(dx=dx,dy=dy))
    for _ in range(40): step(act())
    h0=H(); hit=None
    for _ in range(300):
        step(act(dy=-0.005))
        if np.abs(H()-h0).max()>2e-4: hit=rget(obs,'y'); break
    t=(tx-hx)/u[0]
    pred_center = hy+t*u[1]-0.0533*nv[1]
    print("x",tx,"contact_y",None if hit is None else round(hit,4),"=> top",None if hit is None else round(hit-0.27,4),"model top",round(pred_center+0.0533,4))
    for _ in range(15): step(act(dy=0.05))
env.close()

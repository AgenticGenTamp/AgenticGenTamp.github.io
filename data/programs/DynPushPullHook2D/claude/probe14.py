from env_client import make_env
from ctl import act, rget, oget
import numpy as np
env=make_env()
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def H(obs): return np.array([oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')])
# fine measure of bar top and bottom at x=2.0, seed 42
def measure(x0, ydir, ystart):
    obs,info=env.reset(seed=42)
    def step(a):
        nonlocal obs
        obs,_,_,_,_=env.step(a)
    for _ in range(120):
        d=wrap(np.pi/2*(-ydir)-rget(obs,'theta'))   # arm points away from motion
        if abs(d)<0.005: break
        step(act(dth=d,da=-0.1,dg=-0.02))
    for _ in range(400):
        dx=np.clip(x0-rget(obs,'x'),-0.05,0.05); dy=np.clip(ystart-rget(obs,'y'),-0.05,0.05)
        if abs(dx)<0.002 and abs(dy)<0.002: break
        step(act(dx=dx,dy=dy))
    h0=H(obs)
    for _ in range(400):
        step(act(dy=0.002*ydir))
        if np.abs(H(obs)-h0).max()>1e-5:
            return rget(obs,'y')
    return None
top=measure(2.0,-1,1.30); bot=measure(2.0,+1,0.30)
print("at x=2.0: top",top-0.24,"bot",bot+0.24,"center",(top+bot)/2,"thick",(top-0.24)-(bot+0.24))

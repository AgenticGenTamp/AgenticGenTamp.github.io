from env_client import make_env
from ctl import act, rget, oget
import numpy as np
env=make_env()
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
for seed in [42,0,7]:
    obs,info=env.reset(seed=seed)
    def step(a):
        global obs
        obs,_,_,_,_=env.step(a)
    def H(): return (round(oget(obs,'hook','x'),3),round(oget(obs,'hook','y'),3),round(oget(obs,'hook','theta'),3),oget(obs,'hook','held'))
    hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
    u=np.array([np.cos(hth),np.sin(hth)]); n=np.array([-np.sin(hth),np.cos(hth)])
    P=np.array([hx,hy])-1.4*u-0.0533*n
    base=P-0.594*u
    for _ in range(60): step(act(dth=wrap(hth-rget(obs,'theta')),dg=0.02,da=-0.1))
    for _ in range(400):
        dx=np.clip(base[0]-rget(obs,'x'),-0.05,0.05); dy=np.clip(0.26-rget(obs,'y'),-0.05,0.05)
        if abs(dx)<0.003 and abs(dy)<0.003: break
        step(act(dx=dx,dy=dy))
    for _ in range(400):
        dy=np.clip(base[1]-rget(obs,'y'),-0.05,0.05)
        if abs(dy)<0.003: break
        step(act(dy=dy))
    for _ in range(30): step(act(da=0.1))
    print(seed,"pre-close hook",H())
    for i in range(15):
        step(act(dg=-0.02))
        if oget(obs,'hook','held')>0.5:
            print("  HELD at gap",round(rget(obs,'finger_gap'),3),H()); break
    else:
        print("  not held, gap",round(rget(obs,'finger_gap'),3),H())
    for _ in range(10): step(act(dx=-0.05,dy=0.03))
    print("  after move: robot",round(rget(obs,'x'),3),round(rget(obs,'y'),3),"hook",H())
env.close()

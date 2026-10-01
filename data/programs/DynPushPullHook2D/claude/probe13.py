from env_client import make_env
from ctl import act, rget, oget
import numpy as np
env=make_env()
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
for seed in [42,0,7]:
  for OFF in [0.594]:
    obs,info=env.reset(seed=seed)
    def step(a):
        global obs
        obs,_,_,_,_=env.step(a)
    def H(): return (round(oget(obs,'hook','x'),3),round(oget(obs,'hook','y'),3),round(oget(obs,'hook','theta'),3),oget(obs,'hook','held'))
    hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
    u=np.array([np.cos(hth),np.sin(hth)])
    fe=np.array([hx,hy])-1.4*u
    base=fe-OFF*u
    for _ in range(60):
        step(act(dth=wrap(hth-rget(obs,'theta')),dg=0.02,da=-0.1))
    for _ in range(400):
        dx=np.clip(base[0]-rget(obs,'x'),-0.05,0.05); dy=np.clip(0.26-rget(obs,'y'),-0.05,0.05)
        if abs(dx)<0.003 and abs(dy)<0.003: break
        step(act(dx=dx,dy=dy))
    for _ in range(400):
        dy=np.clip(base[1]-rget(obs,'y'),-0.05,0.05)
        if abs(dy)<0.003: break
        step(act(dy=dy))
    print(seed,OFF,"pos",round(rget(obs,'x'),3),round(rget(obs,'y'),3),"hook",H())
    for _ in range(30): step(act(da=0.1))
    print("  after extend hook",H(),"armj",round(rget(obs,'arm_joint'),3))
    for _ in range(30): step(act(dg=-0.02))
    print("  after close hook",H(),"gap",round(rget(obs,'finger_gap'),3))
    for _ in range(10): step(act(dx=-0.05))
    print("  after pull hook",H(),"robot",round(rget(obs,'x'),3),round(rget(obs,'y'),3))
env.close()

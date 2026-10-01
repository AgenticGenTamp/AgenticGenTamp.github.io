from env_client import make_env
from ctl import act, rget, oget
import numpy as np
env=make_env(); obs,info=env.reset(seed=42)
def step(a):
    global obs
    obs,_,t,_,_=env.step(a); return t
# push the hook with the robot base to find its extent: sweep the robot upward at various x with arm retracted
# instead: move robot far right-bottom then move up slowly, printing hook pose changes
def P(): return (round(rget(obs,'x'),3),round(rget(obs,'y'),3))
def H(): return (round(oget(obs,'hook','x'),3),round(oget(obs,'hook','y'),3),round(oget(obs,'hook','theta'),3))
print("hook0",H())
# retract arm, close gripper to make robot compact
for _ in range(30): step(act(da=-0.1,dg=-0.02))
for tx in [1.6,2.0,2.4,2.8,3.2]:
    obs,info=env.reset(seed=42)
    for _ in range(30): step(act(da=-0.1,dg=-0.02))
    for _ in range(200):
        if abs(rget(obs,'x')-tx)<0.005: break
        step(act(dx=np.clip(tx-rget(obs,'x'),-0.05,0.05)))
    h0=H()
    for _ in range(60):
        step(act(dy=0.05))
        if H()!=h0:
            print("x",tx,"contact at robot y",round(rget(obs,'y'),3),"hook",H()); break
    else:
        print("x",tx,"no contact, final y",round(rget(obs,'y'),3))
env.close()

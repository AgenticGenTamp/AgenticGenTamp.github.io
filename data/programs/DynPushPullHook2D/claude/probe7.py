from env_client import make_env
from ctl import act, rget, oget
import numpy as np
env=make_env(); obs,info=env.reset(seed=42)
def step(a):
    global obs
    obs,_,t,_,_=env.step(a); return t
def state():
    return rget(obs,'x'),rget(obs,'y'),rget(obs,'theta'),rget(obs,'arm_joint'),rget(obs,'finger_gap')
hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
print("hook",hx,hy,hth)
# grasp point on long bar
gp=np.array([hx,hy])+1.25*np.array([-np.cos(hth),-np.sin(hth)])
rth=hth
base=gp-0.40*np.array([np.cos(rth),np.sin(rth)])
print("target base",base,"rth",rth)
# open gripper
for _ in range(30): step(act(dg=0.02))
# rotate
for _ in range(200):
    x,y,th,aj,fg=state()
    d=(rth-th+np.pi)%(2*np.pi)-np.pi
    if abs(d)<0.005: break
    step(act(dth=d))
# move x to base[0] at low y first
for _ in range(400):
    x,y,th,aj,fg=state()
    if abs(x-base[0])<0.005 and abs(y-0.35)<0.005: break
    step(act(dx=np.clip(base[0]-x,-0.05,0.05), dy=np.clip(0.35-y,-0.05,0.05)))
print("pos1",state())
for _ in range(400):
    x,y,th,aj,fg=state()
    if abs(y-base[1])<0.005: break
    step(act(dy=np.clip(base[1]-y,-0.05,0.05)))
print("pos2",state())
print("hook now",oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta'),oget(obs,'hook','held'))
for _ in range(30): step(act(dg=-0.02))
print("after close",state(),"held",oget(obs,'hook','held'))
# move robot and see if hook follows
for _ in range(20): step(act(dx=-0.05))
print("robot",state(),"hook",oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta'),oget(obs,'hook','held'))
env.close()

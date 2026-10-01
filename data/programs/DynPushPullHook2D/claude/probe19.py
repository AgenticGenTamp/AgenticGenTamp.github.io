from env_client import make_env
from ctl import act, rget, oget
from grasp import grasp_seq, wrap
import numpy as np
env=make_env()
obs,info=env.reset(seed=42)
obs,n,ok=grasp_seq(env,obs)
print("grasped",ok,n)
term=False
def step(a):
    global obs,term,n
    obs,r,t,tr,i=env.step(a); term=term or t; n+=1
def R(f): return rget(obs,f)
def H(f): return oget(obs,'hook',f)
def B(f): return oget(obs,'target_block',f)
print("hook after grasp",round(H('x'),3),round(H('y'),3),round(H('theta'),3))
# rotate to pi/2
for _ in range(200):
    d=wrap(np.pi/2-R('theta'))
    if abs(d)<0.004: break
    step(act(dth=d))
print("after rot: robot",round(R('x'),3),round(R('y'),3),round(R('theta'),3),"hook",round(H('x'),3),round(H('y'),3),round(H('theta'),3),"held",H('held'))
bx,by=B('x'),B('y')
tgt=np.array([bx-0.35, by+0.2*1.15+0.06-1.95])
print("target base",tgt, "block",bx,by)
for _ in range(400):
    dx=np.clip(tgt[0]-R('x'),-0.05,0.05); dy=np.clip(tgt[1]-R('y'),-0.05,0.05)
    if abs(dx)<0.003 and abs(dy)<0.003: break
    step(act(dx=dx,dy=dy))
print("at",round(R('x'),3),round(R('y'),3),"hook",round(H('x'),3),round(H('y'),3),"block",round(B('x'),3),round(B('y'),3),"term",term,"steps",n)
for _ in range(60):
    step(act(dy=-0.05))
    if term: break
print("after pull: block",round(B('x'),3),round(B('y'),3),"robot y",round(R('y'),3),"term",term,"steps",n)
env.close()

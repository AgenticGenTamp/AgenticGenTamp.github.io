import numpy as np, fk, ctrl
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=6)
def g(n,f): return float(obs.get(obs.get_object_from_name(n),f))
def J(): return np.array([g("robot","pos_arm_joint%d"%i) for i in range(1,8)])
def B(): return np.array([g("robot","pos_base_x"),g("robot","pos_base_y"),g("robot","pos_base_rot")])
def C(): return np.array([g("cube_0","x"),g("cube_0","y")])
def step(tq,tb=None,grip=0.0):
    global obs
    a=ctrl.action(J(),tq,B(),tb,grip); obs,rew,t,tr,i2=env.step(a)
yaw=B()[2]; fwd=np.array([np.cos(yaw),np.sin(yaw)]); left=np.array([-np.sin(yaw),np.cos(yaw)])
q=ctrl.ik_local([0.45,0.0,-0.02],J())
for _ in range(140): step(q)
print("jerr",round(float(np.max(np.abs(J()-q))),3))
def offvec():
    p,_=fk.fk(J(),*B()); return p[:2]-B()[:2]
def run(d):
    c=C(); ov=offvec()
    start=c-d*fwd-0.32*left-ov
    for _ in range(60): step(q,np.array([start[0],start[1],yaw]))
    c=C(); moved=0.0
    for k in range(22):
        tb=np.array([B()[0]+left[0]*0.06, B()[1]+left[1]*0.06, yaw]); step(q,tb)
        m=np.linalg.norm(C()-c)
        if m>moved: moved=m
    print("d",round(d,2),"cube moved",round(float(moved),3),"cube",np.round(C(),3))
for d in [0.0,0.06,0.12,0.18,0.24,0.30,0.36]:
    run(d)
env.close()

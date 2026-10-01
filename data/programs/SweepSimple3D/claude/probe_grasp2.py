import numpy as np, fk, ctrl
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=6)
def g(n,f): return float(obs.get(obs.get_object_from_name(n),f))
def J(): return np.array([g("robot","pos_arm_joint%d"%i) for i in range(1,8)])
def B(): return np.array([g("robot","pos_base_x"),g("robot","pos_base_y"),g("robot","pos_base_rot")])
def C(): return np.array([g("cube_0","x"),g("cube_0","y"),g("cube_0","z")])
def step(tq,tb=None,grip=0.0):
    global obs
    a=ctrl.action(J(),tq,B(),tb,grip); obs,rew,t,tr,i2=env.step(a)
yaw=B()[2]
qh=ctrl.ik_local([0.45,0.0,0.20],J())
for _ in range(140): step(qh,None,0.0)
def offvec():
    p,_=fk.fk(J(),*B()); return p[:2]-B()[:2]
def attempt(zl):
    c=C(); ov=offvec()
    tb=np.array([c[0]-ov[0], c[1]-ov[1], yaw])
    for _ in range(70): step(qh,tb,0.0)
    p,_=fk.fk(J(),*B()); print(" aligned tip",np.round(p,3),"cube",np.round(C(),3))
    ql=ctrl.ik_local([0.45,0.0,zl],J())
    for _ in range(50): step(ql,tb,0.0)
    print(" low jerr",round(float(np.max(np.abs(J()-ql))),3),"cube",np.round(C(),3))
    for _ in range(8): step(ql,tb,1.0)
    for _ in range(60): step(qh,tb,1.0)
    c2=C(); p,_=fk.fk(J(),*B())
    print(" ZL",zl,"cube after lift",np.round(c2,3),"tip",np.round(p,3),"GRASPED" if c2[2]>0.06 else "no")
    for _ in range(8): step(qh,tb,0.0)
    for _ in range(30): step(qh,tb,0.0)
for zl in [0.0,-0.03,0.03]:
    attempt(zl)
env.close()

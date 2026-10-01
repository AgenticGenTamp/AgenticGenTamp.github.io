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
def tip():
    p,_=fk.fk(J(),*B()); return p
yaw=B()[2]; fwd=np.array([np.cos(yaw),np.sin(yaw)])
print("fwd",np.round(fwd,3))

def attempt(off, zlow=-0.02):
    c=C()
    want_tip_xy=c[:2]-off*fwd
    # config: local tip (0.45,0,zhigh) then descend
    qh=ctrl.ik_local([0.45,0.0,0.18],J())
    for _ in range(100): step(qh,None,0.0)
    p=tip(); d=p[:2]-B()[:2]
    tb=np.array([want_tip_xy[0]-d[0], want_tip_xy[1]-d[1], yaw])
    for _ in range(90): step(qh,tb,0.0)
    print(" pos: tip",np.round(tip(),3),"want",np.round(want_tip_xy,3),"cube",np.round(C(),3))
    ql=ctrl.ik_local([0.45,0.0,zlow],J())
    for _ in range(60): step(ql,tb,0.0)
    print(" low: tip",np.round(tip(),3),"jerr",round(float(np.max(np.abs(J()-ql))),3))
    for _ in range(6): step(ql,tb,1.0)
    for _ in range(80): step(qh,tb,1.0)
    print(" OFF",off,"after lift cube",np.round(C(),3),"tip",np.round(tip(),3),"=> resid(cube-tip)",np.round(C()-tip(),3))
    # release
    for _ in range(6): step(qh,tb,0.0)

for off in [0.19,0.10,0.28]:
    attempt(off)
env.close()

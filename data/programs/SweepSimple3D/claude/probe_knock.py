import numpy as np, fk, ctrl
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=6)
def g(n,f): return float(obs.get(obs.get_object_from_name(n),f))
def J(): return np.array([g("robot","pos_arm_joint%d"%i) for i in range(1,8)])
def B(): return np.array([g("robot","pos_base_x"),g("robot","pos_base_y"),g("robot","pos_base_rot")])
def W(): return np.array([g("wiper_0","x"),g("wiper_0","y"),g("wiper_0","z"),g("wiper_0","qw"),g("wiper_0","qx"),g("wiper_0","qy")])
def step(tq,tb=None,grip=0.0):
    global obs
    a=ctrl.action(J(),tq,B(),tb,grip); obs,rew,t,tr,i2=env.step(a)
yaw=B()[2]; fwd=np.array([np.cos(yaw),np.sin(yaw)]); left=np.array([-np.sin(yaw),np.cos(yaw)])
w0=W(); print("wiper",np.round(w0,3),"base",np.round(B(),3),flush=True)
qh=ctrl.ik_local([0.45,0.0,0.55],J())
for _ in range(150): step(qh)
print("high jerr",round(float(np.max(np.abs(J()-qh))),3),flush=True)
def offv():
    p,_=fk.fk(J(),*B()); return p[:2]-B()[:2]
def attempt(fo,lo):
    w=W()
    ov=offv(); want=w[:2]+fo*fwd+lo*left
    tb=np.array([want[0]-ov[0], want[1]-ov[1], yaw])
    for _ in range(80): step(qh,tb)
    ql=ctrl.ik_local([0.45,0.0,0.10],J())
    for _ in range(70): step(ql,tb)
    w2=W()
    moved=np.linalg.norm(w2[:2]-w[:2])+abs(w2[3]-w[3])
    print("fo",fo,"lo",lo,"moved",round(float(moved),3),"wiper",np.round(w2,3),"jerr",round(float(np.max(np.abs(J()-ql))),3),flush=True)
    for _ in range(70): step(qh,tb)
for (fo,lo) in [(0,0),(0.25,0),(-0.25,0),(0,0.25),(0,-0.25)]:
    attempt(fo,lo)
env.close()

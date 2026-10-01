import numpy as np, fk, ctrl
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=6)
def g(n,f): return float(obs.get(obs.get_object_from_name(n),f))
def J(): return np.array([g("robot","pos_arm_joint%d"%i) for i in range(1,8)])
def B(): return np.array([g("robot","pos_base_x"),g("robot","pos_base_y"),g("robot","pos_base_rot")])
def W(): return np.array([g("wiper_0","x"),g("wiper_0","y"),g("wiper_0","z"),g("wiper_0","qw")])
def step(tq,tb=None,grip=0.0):
    global obs
    a=ctrl.action(J(),tq,B(),tb,grip); obs,rew,t,tr,i2=env.step(a)
w=W(); b0=B(); print("wiper",np.round(w,3),"base",np.round(b0,3))
q=ctrl.ik_local([0.45,0.0,0.15],J())
print("q",np.round(q,3))
for _ in range(120): step(q)
print("jerr",round(float(np.max(np.abs(J()-q))),3))
# move base so that distance to wiper = 0.45 along the current base->wiper direction
d=w[:2]-B()[:2]; d/=np.linalg.norm(d)
tb=np.array([w[0]-0.45*d[0], w[1]-0.45*d[1], b0[2]])
for _ in range(90): step(q,tb)
print("base now",np.round(B(),3),"dist",round(float(np.linalg.norm(W()[:2]-B()[:2])),3),"wiper",np.round(W(),3))
azim_world=np.arctan2(d[1],d[0]); print("world azim base->wiper",round(float(azim_world),3),"base yaw",round(float(B()[2]),3))
# sweep joint1
w0=W()
for k in range(260):
    tq=J().copy(); tq[0]=J()[0]+0.05
    step(tq,tb)
    w1=W()
    if np.linalg.norm(w1[:2]-w0[:2])>0.01 or abs(w1[3]-w0[3])>0.03:
        p,_=fk.fk(J(),*B())
        print("HIT at joint1",round(float(J()[0]),3),"wiper",np.round(w1,3),"predtip",np.round(p,3))
        break
else:
    print("no hit; joint1",round(float(J()[0]),3),"wiper",np.round(W(),3))
env.close()

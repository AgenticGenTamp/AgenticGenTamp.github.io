import numpy as np
from env_client import make_env
TGT=np.array([0.0,1.8838,3.1416,-1.5828,0.0,0.3250,1.5708])  # f30_d44
env=make_env(); obs,info=env.reset(seed=1)
R=lambda f: float(obs.get(obs.get_object_from_name("robot"),f))
C=lambda f: float(obs.get(obs.get_object_from_name("cube_0"),f))
J=lambda: np.array([R("pos_arm_joint%d"%i) for i in range(1,8)])
pose=lambda: np.array([R("pos_base_x"),R("pos_base_y"),R("pos_base_rot")])
cube=lambda: np.array([C("x"),C("y"),C("z")])
def step(a):
    global obs
    obs,rew,term,trunc,info=env.step(np.asarray(a,dtype=np.float32))
def act(vx=0.,vy=0.,vw=0.):
    a=np.zeros(11); a[0]=vx; a[1]=vy; a[2]=vw
    a[3:10]=np.clip(TGT-J(),-0.1,0.1); return a
for i in range(115): step(act())
print("settled err",np.round(J()-TGT,3).tolist())
YAW=-0.20
for i in range(40):
    e=YAW-pose()[2]
    if abs(e)<0.005: break
    step(act(vw=float(np.clip(e,-0.1,0.1))))
print("yaw",round(pose()[2],4))
u=np.array([np.cos(pose()[2]),np.sin(pose()[2])]); n=np.array([-u[1],u[0]])
c0=cube()
def moveto(t,rate=0.1,maxn=55):
    for i in range(maxn):
        p=pose(); d=t-p[:2]
        if np.abs(d).max()<0.012: break
        step(act(vx=float(np.clip(d[0]*3,-rate,rate)),vy=float(np.clip(d[1]*3,-rate,rate))))
moveto(c0[:2]-u*0.55)
print("start base",np.round(pose(),3).tolist(),"cube",np.round(c0,4).tolist())
hit=None
for i in range(30):
    step(act(vx=u[0]*0.03,vy=u[1]*0.03))
    c=cube()
    if np.abs(c[:2]-c0[:2]).max()>0.003:
        p=pose(); rel=c0[:2]-p[:2]
        hit=(float(rel@u),float(rel@n))
        print("CONTACT@yaw%.3f fwd=%.4f lat=%.4f"%(p[2],hit[0],hit[1])); break
if hit is None: print("no contact at yaw -0.2")
# fast push
for i in range(14): step(act(vx=u[0]*0.08,vy=u[1]*0.08))
p=pose(); c=cube(); rel=c[:2]-p[:2]
print("after fast push(0.08x14) base",np.round(p,3).tolist(),"cube",np.round(c,4).tolist(),
      "fwd",round(float(rel@u),4),"lat",round(float(rel@n),4),"armerr",np.round(J()-TGT,3).tolist())
# settle in place: does cube stay / arm relax pushing it further?
for i in range(8): step(act())
c=cube(); p=pose(); rel=c[:2]-p[:2]
print("after 8 idle: fwd",round(float(rel@u),4),"lat",round(float(rel@n),4),"cube",np.round(c,4).tolist())
# lateral sweep test
c1=cube()
for i in range(12): step(act(vx=n[0]*0.03,vy=n[1]*0.03))
c=cube(); p=pose(); rel=c[:2]-p[:2]
print("after lateral+0.03x12: cube d=",np.round(c[:2]-c1[:2],4).tolist(),"fwd",round(float(rel@u),4),"lat",round(float(rel@n),4))
# slow precise push to a goal 0.15 further along u
c2=cube(); goal=c2[:2]+u*0.15
for i in range(14): step(act(vx=u[0]*0.03,vy=u[1]*0.03))
for i in range(6): step(act())
c=cube(); print("goal",np.round(goal,4).tolist(),"final cube",np.round(c[:2],4).tolist(),"err",round(float(np.linalg.norm(c[:2]-goal)),4))
env.close()

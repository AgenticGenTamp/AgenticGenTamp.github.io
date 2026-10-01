import numpy as np
from env_client import make_env
TGT=np.array([0.0,1.7939,3.1416,-1.7065,0.0,0.3588,1.5708])  # f30_d40
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
print("settled err",np.round(J()-TGT,4).tolist(),"yaw",round(pose()[2],4))
u=np.array([np.cos(pose()[2]),np.sin(pose()[2])]); n=np.array([-u[1],u[0]])
y0=pose()[2]
for i in range(12): step(act(vx=n[0]*0.05,vy=n[1]*0.05))
print("yaw drift after 0.05 lateral x12:",round(pose()[2]-y0,4))
for i in range(12): step(act(vw=0.05))
print("yaw after 12x vw=0.05:",round(pose()[2],4),"(expect y0+0.6=%.3f)"%(y0+0.6))
for i in range(12): step(act(vw=-0.05))
print("yaw back:",round(pose()[2],4))
u=np.array([np.cos(pose()[2]),np.sin(pose()[2])]); n=np.array([-u[1],u[0]])
c0=cube()
def moveto(t,rate=0.08,maxn=60):
    for i in range(maxn):
        p=pose(); d=t-p[:2]
        if np.abs(d).max()<0.012: break
        step(act(vx=float(np.clip(d[0]*3,-rate,rate)),vy=float(np.clip(d[1]*3,-rate,rate))))
moveto(c0[:2]-u*0.50)
p=pose(); print("start",np.round(p,4).tolist(),"yawdrift",round(p[2]-y0,4))
u=np.array([np.cos(p[2]),np.sin(p[2])]); n=np.array([-u[1],u[0]])
hit=False
for i in range(26):
    step(act(vx=u[0]*0.03,vy=u[1]*0.03))
    c=cube()
    if np.abs(c[:2]-c0[:2]).max()>0.003:
        p=pose(); rel=c0[:2]-p[:2]
        print("CONTACT fwd=%.4f lat=%.4f yaw=%.4f"%(float(rel@u),float(rel@n),p[2])); hit=True; break
if not hit: print("NO CONTACT with d40; base",np.round(pose(),3).tolist())
if hit:
    for rate in (0.03,0.05):
        for i in range(12):
            step(act(vx=u[0]*rate,vy=u[1]*rate))
            if i%4==3:
                p=pose(); c=cube(); rel=c[:2]-p[:2]
                print("  rate",rate,"i",i,"fwd",round(float(rel@u),4),"lat",round(float(rel@n),4),
                      "cubez",round(c[2],4),"yaw",round(p[2],4))
    print("armerr",np.round(J()-TGT,3).tolist())
    # lateral sweep while in contact
    c1=cube()
    for i in range(10): step(act(vx=n[0]*0.03,vy=n[1]*0.03))
    c=cube(); print("lateral sweep cube d=",np.round(c[:2]-c1[:2],4).tolist())
env.close()

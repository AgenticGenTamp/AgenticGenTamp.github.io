import numpy as np
from env_client import make_env
CFG={"f30_d40":[0.0,1.7939,3.1416,-1.7065,0.0,0.3588,1.5708],
     "f30_d44":[0.0,1.8838,3.1416,-1.5828,0.0,0.3250,1.5708]}
env=make_env(); obs,info=env.reset(seed=1)
R=lambda f: float(obs.get(obs.get_object_from_name("robot"),f))
C=lambda f: float(obs.get(obs.get_object_from_name("cube_0"),f))
J=lambda: np.array([R("pos_arm_joint%d"%i) for i in range(1,8)])
pose=lambda: np.array([R("pos_base_x"),R("pos_base_y"),R("pos_base_rot")])
cube=lambda: np.array([C("x"),C("y"),C("z")])
def step(a):
    global obs
    obs,rew,term,trunc,info=env.step(np.asarray(a,dtype=np.float32))
TGT=np.array(CFG["f30_d44"])
def act(vx=0.,vy=0.,vw=0.):
    a=np.zeros(11); a[0]=vx; a[1]=vy; a[2]=vw
    a[3:10]=np.clip(TGT-J(),-0.1,0.1); return a
for i in range(45): step(act())
print("t45 err",np.round(J()-TGT,3).tolist())
for i in range(45): step(act())
print("t90 err",np.round(J()-TGT,3).tolist())
for i in range(30): step(act())
print("t120 err",np.round(J()-TGT,4).tolist())
c0=cube(); print("cube0",np.round(c0,4).tolist(),"base",np.round(pose(),3).tolist())
def moveto(t,rate=0.1,maxn=50):
    for i in range(maxn):
        p=pose(); d=t-p[:2]
        if np.abs(d).max()<0.012: break
        step(act(vx=np.clip(d[0]*3,-rate,rate),vy=np.clip(d[1]*3,-rate,rate)))
def spin(radius,nmax=80,rate=0.08):
    global c0
    c0=cube()
    moveto(c0[:2]+np.array([0.0,radius]))
    p=pose(); r=np.linalg.norm(c0[:2]-p[:2])
    print("spin r=%.3f base"%r,np.round(p,3).tolist())
    for i in range(nmax):
        step(act(vw=rate))
        c=cube()
        if np.abs(c[:2]-c0[:2]).max()>0.003:
            p=pose(); rel=c0[:2]-p[:2]; yaw=p[2]
            ang=np.arctan2(rel[1],rel[0])-yaw
            ang=(ang+np.pi)%(2*np.pi)-np.pi
            print("  CONTACT yaw=%.4f rel=%s dist=%.4f angle_in_base=%.4f"%(yaw,np.round(rel,4).tolist(),np.linalg.norm(rel),ang))
            print("  fwd=%.4f lat=%.4f cube_now=%s"%(np.linalg.norm(rel)*np.cos(ang),np.linalg.norm(rel)*np.sin(ang),np.round(c,4).tolist()))
            return yaw,np.linalg.norm(rel),ang
    print("  no contact; yaw now",round(pose()[2],3))
    return None
r=spin(0.30)
if r is None: r=spin(0.42)
if r is None: r=spin(0.22)
if r is not None:
    yaw,dist,ang=r
    # straight push along current tip direction
    c=cube(); u=np.array([np.cos(pose()[2]),np.sin(pose()[2])]); n=np.array([-u[1],u[0]])
    print("PUSH TEST u",np.round(u,3).tolist())
    for i in range(20):
        step(act(vx=u[0]*0.03,vy=u[1]*0.03))
        if i%4==0:
            p=pose(); c=cube(); rel=c[:2]-p[:2]
            print("  i",i,"base",np.round(p,4).tolist(),"cube",np.round(c,4).tolist(),
                  "fwd",round(float(rel@u),4),"lat",round(float(rel@n),4))
    print("arm err after push",np.round(J()-TGT,3).tolist())
env.close()

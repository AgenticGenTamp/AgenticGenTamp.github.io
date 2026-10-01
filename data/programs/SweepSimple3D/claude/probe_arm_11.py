import numpy as np
from env_client import make_env
TGT=np.array([0.0,1.8838,3.1416,-1.5828,0.0,0.3250,1.5708])  # f30_d44
GRIP=0.0
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
    a[3:10]=np.clip(TGT-J(),-0.1,0.1); a[10]=GRIP; return a
for i in range(130): step(act())
print("settled err",np.round(J()-TGT,4).tolist(),"grip",round(R("pos_gripper"),3),"yaw",round(pose()[2],4))
def uv():
    y=pose()[2]; u=np.array([np.cos(y),np.sin(y)]); return u,np.array([-u[1],u[0]])
def moveto(t,rate=0.06,maxn=70):
    for i in range(maxn):
        p=pose(); d=t-p[:2]
        if np.abs(d).max()<0.006: break
        step(act(vx=float(np.clip(d[0]*4,-rate,rate)),vy=float(np.clip(d[1]*4,-rate,rate))))
for latoff in (-0.04,0.0,-0.08):
    u,n=uv(); c=cube()
    for i in range(9): step(act(vx=-u[0]*0.06,vy=-u[1]*0.06))
    moveto(c[:2]-u*0.45-n*latoff)
    u,n=uv(); c0=cube(); p=pose(); rel0=c0[:2]-p[:2]; y0=p[2]
    print("=== latoff",latoff,"start fwd %.3f lat %.3f yaw %.4f"%(float(rel0@u),float(rel0@n),y0))
    hit=None
    for i in range(24):
        step(act(vx=u[0]*0.03,vy=u[1]*0.03))
        cc=cube()
        if np.linalg.norm(cc[:2]-c0[:2])>0.006:
            p=pose(); rel=c0[:2]-p[:2]; hit=(float(rel@u),float(rel@n))
            print("  CONTACT fwd=%.4f lat=%.4f"%hit); break
    if hit is None:
        print("  no contact (moved %.4f)"%np.linalg.norm(cube()[:2]-c0[:2])); continue
    for i in range(24):
        step(act(vx=u[0]*0.03,vy=u[1]*0.03))
        if i%6==5:
            p=pose(); cc=cube(); rel=cc[:2]-p[:2]
            print("   push i%d fwd %.4f lat %.4f z %.4f dyaw %.4f"%(i,float(rel@u),float(rel@n),cc[2],p[2]-y0))
    cnet=cube()[:2]-c0[:2]; bnet=pose()[:2]-(c0[:2]-np.array([rel0@u*u[0]+rel0@n*n[0],0])*0)
    print("   cube net",np.round(cnet,4).tolist(),"|d|=%.3f"%np.linalg.norm(cnet),
          "along_u=%.3f"%float(cnet@u),"lat=%.3f"%float(cnet@n),"armerr",np.round(J()-TGT,3).tolist())
env.close()

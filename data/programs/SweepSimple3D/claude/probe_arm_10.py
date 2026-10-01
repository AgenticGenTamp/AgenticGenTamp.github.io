import numpy as np
from env_client import make_env
TGT=np.array([0.0,1.7939,3.1416,-1.7065,0.0,0.3588,1.5708])  # f30_d40
GRIP=1.0
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
for i in range(115): step(act())
print("settled err",np.round(J()-TGT,4).tolist(),"grip",round(R("pos_gripper"),4),"yaw",round(pose()[2],4))
def uv():
    y=pose()[2]; u=np.array([np.cos(y),np.sin(y)]); return u,np.array([-u[1],u[0]])
def moveto(t,rate=0.08,maxn=60):
    for i in range(maxn):
        p=pose(); d=t-p[:2]
        if np.abs(d).max()<0.012: break
        step(act(vx=float(np.clip(d[0]*3,-rate,rate)),vy=float(np.clip(d[1]*3,-rate,rate))))
for latoff in (0.0,-0.03,0.03):
    u,n=uv(); c=cube()
    for i in range(10): step(act(vx=-u[0]*0.08,vy=-u[1]*0.08))
    moveto(c[:2]-u*0.42-n*latoff)
    u,n=uv(); c0=cube(); p=pose(); rel0=c0[:2]-p[:2]
    print("=== latoff",latoff,"start fwd %.3f lat %.3f"%(float(rel0@u),float(rel0@n)))
    hit=None
    for i in range(22):
        step(act(vx=u[0]*0.03,vy=u[1]*0.03))
        cc=cube()
        if np.linalg.norm(cc[:2]-c0[:2])>0.006:
            p=pose(); rel=c0[:2]-p[:2]
            hit=(float(rel@u),float(rel@n))
            print("  CONTACT fwd=%.4f lat=%.4f"%hit); break
    if hit is None:
        print("  no contact (cube moved %.4f)"%np.linalg.norm(cube()[:2]-c0[:2])); continue
    for i in range(12):
        step(act(vx=u[0]*0.03,vy=u[1]*0.03))
        if i%4==3:
            p=pose(); cc=cube(); rel=cc[:2]-p[:2]
            print("   push i%d fwd %.4f lat %.4f z %.4f"%(i,float(rel@u),float(rel@n),cc[2]))
    print("   cube net",np.round(cube()[:2]-c0[:2],4).tolist(),"|d|=%.3f"%np.linalg.norm(cube()[:2]-c0[:2]),
          "armerr",np.round(J()-TGT,3).tolist())
env.close()

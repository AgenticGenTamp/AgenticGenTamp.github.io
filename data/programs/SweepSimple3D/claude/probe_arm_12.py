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
def act(vx=0.,vy=0.,vw=0.,g=0.0):
    a=np.zeros(11); a[0]=vx; a[1]=vy; a[2]=vw
    a[3:10]=np.clip(TGT-J(),-0.1,0.1); a[10]=g; return a
for i in range(120): step(act())
print("settled err",np.round(J()-TGT,4).tolist(),"yaw",round(pose()[2],4))
def uv():
    y=pose()[2]; u=np.array([np.cos(y),np.sin(y)]); return u,np.array([-u[1],u[0]])
def rel():
    u,n=uv(); d=cube()[:2]-pose()[:2]; return float(d@u),float(d@n)
def goto_rel(fwd,lat,maxn=70,rate=0.06):
    for i in range(maxn):
        f,l=rel(); u,n=uv()
        e=(fwd-f)*u+(lat-l)*n
        if np.linalg.norm(e)<0.005: break
        v=np.clip(e*4,-rate,rate); step(act(vx=float(v[0]),vy=float(v[1])))
    return rel()
def sweep(fwd,lat0,direction,nmax=12):
    f,l=goto_rel(fwd,lat0)
    print("=== sweep fwd %.3f lat %.3f (want %.2f/%.2f) yaw %.3f"%(f,l,fwd,lat0,pose()[2]))
    c0=cube()
    for i in range(nmax):
        u,n=uv(); step(act(vx=n[0]*0.03*direction,vy=n[1]*0.03*direction))
        cc=cube()
        if np.linalg.norm(cc[:2]-c0[:2])>0.004:
            f,l=rel()
            print("   NUDGE at cube_lat %.4f cube_fwd %.4f (moved %.4f)"%(l,f,np.linalg.norm(cc[:2]-c0[:2])))
            return
    print("   no nudge; end cube_lat %.4f"%rel()[1])
sweep(0.30, 0.12, +1)
sweep(0.34, -0.12, -1)
sweep(0.26, 0.12, +1)
env.close()

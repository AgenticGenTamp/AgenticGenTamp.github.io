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
def act(vx=0.,vy=0.):
    a=np.zeros(11); a[0]=vx; a[1]=vy
    a[3:10]=np.clip(TGT-J(),-0.1,0.1); return a
for i in range(120): step(act())
print("settled err",np.round(J()-TGT,3).tolist(),"yaw",round(pose()[2],4))
def uv():
    y=pose()[2]; u=np.array([np.cos(y),np.sin(y)]); return u,np.array([-u[1],u[0]])
def rel():
    u,n=uv(); d=cube()[:2]-pose()[:2]; return float(d@u),float(d@n)
def moveto(t,rate=0.06,maxn=60):
    for i in range(maxn):
        p=pose(); d=t-p[:2]
        if np.abs(d).max()<0.006: break
        step(act(vx=float(np.clip(d[0]*4,-rate,rate)),vy=float(np.clip(d[1]*4,-rate,rate))))
def sweep(fwd,lat0,nmax=13):
    u,n=uv(); c=cube()
    moveto(c[:2]-u*0.60-n*lat0)              # stand off behind
    u,n=uv(); c=cube()
    moveto(c[:2]-u*fwd-n*lat0)               # slide in to sweep radius
    f,l=rel(); print("=== sweep want fwd%.2f lat%.2f -> got fwd %.3f lat %.3f yaw %.3f"%(fwd,lat0,f,l,pose()[2]))
    c0=cube()
    for i in range(nmax):
        u,n=uv(); step(act(vx=n[0]*0.03,vy=n[1]*0.03))
        cc=cube()
        if np.linalg.norm(cc[:2]-c0[:2])>0.004:
            f,l=rel(); print("   NUDGE cube_lat %.4f cube_fwd %.4f moved %.4f"%(l,f,np.linalg.norm(cc[:2]-c0[:2]))); return
    print("   no nudge; end cube_lat %.4f fwd %.4f"%rel()[::-1][::-1])
sweep(0.30,0.13)
sweep(0.34,0.13)
sweep(0.26,0.13)
env.close()

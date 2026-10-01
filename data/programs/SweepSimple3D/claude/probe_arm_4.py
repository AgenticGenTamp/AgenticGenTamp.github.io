import time, numpy as np
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=1)
R=lambda f: float(obs.get(obs.get_object_from_name("robot"),f))
C=lambda f: float(obs.get(obs.get_object_from_name("cube_0"),f))
def J(): return np.array([R("pos_arm_joint%d"%i) for i in range(1,8)])
def step(a):
    global obs
    obs,rew,term,trunc,info=env.step(np.asarray(a,dtype=np.float32))
TGT=np.array([0,0.6,3.142,-1.6,0,-1.2,1.571])
def act(bx=0.,by=0.,bw=0.,grip=0.):
    a=np.zeros(11); a[0]=bx; a[1]=by; a[2]=bw
    a[3:10]=np.clip(TGT-J(),-0.1,0.1); a[10]=grip
    return a
def pose(): return R("pos_base_x"),R("pos_base_y"),R("pos_base_rot")
def cube(): return C("x"),C("y"),C("z")
# settle arm
for i in range(35): step(act())
print("arm settled",np.round(J(),3).tolist())
cx0,cy0,cz0=cube(); print("cube0",round(cx0,4),round(cy0,4),round(cz0,4),"pose",np.round(pose(),3).tolist())
def moveto(tx,ty,rate=0.1,maxn=40):
    for i in range(maxn):
        x,y,w=pose()
        dx=np.clip((tx-x)*3,-rate,rate); dy=np.clip((ty-y)*3,-rate,rate)
        if abs(tx-x)<0.01 and abs(ty-y)<0.01: break
        step(act(bx=dx,by=dy))
    return pose()
def scan(dirvec,nmax,rate=0.03):
    """drive slowly; return first pose where cube moved >3mm"""
    global cx0,cy0
    for i in range(nmax):
        step(act(bx=dirvec[0]*rate,by=dirvec[1]*rate))
        x,y,z=cube()
        if abs(x-cx0)>0.003 or abs(y-cy0)>0.003:
            p=pose()
            print("  CONTACT at base",np.round(p,4).tolist(),"cube",round(x,4),round(y,4),round(z,4),
                  "dcube",round(x-cx0,4),round(y-cy0,4))
            return p
    print("  no contact after",nmax,"steps; base",np.round(pose(),3).tolist())
    return None
# pass 1: align base x with cube x, come from +y
for xoff in (0.0,0.15,-0.15,0.30):
    print("=== xoff",xoff)
    moveto(cx0+xoff, cy0+0.55, rate=0.1, maxn=45)
    print("  start",np.round(pose(),3).tolist())
    p=scan((0,-1), 26)
    if p is not None:
        # keep pushing 10 more steps to see tracking
        for i in range(10): step(act(by=-0.03))
        print("  after push: base",np.round(pose(),4).tolist(),"cube",np.round(cube(),4).tolist())
        break
    moveto(cx0+xoff, cy0+0.55, rate=0.1, maxn=45)
env.close()

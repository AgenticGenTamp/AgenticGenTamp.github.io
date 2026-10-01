import time, numpy as np
from env_client import make_env
LADDER = {
 "f30_d24":[0.0,1.4162,3.1416,-2.1027,0.0,0.3773,1.5708],
 "f30_d28":[0.0,1.5161,3.1416,-2.0179,0.0,0.3923,1.5708],
 "f30_d32":[0.0,1.6114,3.1416,-1.9235,0.0,0.3933,1.5708],
 "f30_d36":[0.0,1.7036,3.1416,-1.8197,0.0,0.3818,1.5708],
 "f30_d40":[0.0,1.7939,3.1416,-1.7065,0.0,0.3588,1.5708],
 "f30_d44":[0.0,1.8838,3.1416,-1.5828,0.0,0.3250,1.5708],
 "f30_d48":[0.0,1.9747,3.1416,-1.4472,0.0,0.2803,1.5708],
}
env=make_env(); obs,info=env.reset(seed=1)
R=lambda f: float(obs.get(obs.get_object_from_name("robot"),f))
C=lambda f: float(obs.get(obs.get_object_from_name("cube_0"),f))
def J(): return np.array([R("pos_arm_joint%d"%i) for i in range(1,8)])
def pose(): return np.array([R("pos_base_x"),R("pos_base_y"),R("pos_base_rot")])
def cube(): return np.array([C("x"),C("y"),C("z")])
def step(a):
    global obs
    obs,rew,term,trunc,info=env.step(np.asarray(a,dtype=np.float32))
TGT=np.array(LADDER["f30_d24"])
def act(vx=0.,vy=0.,vw=0.,grip=0.):
    a=np.zeros(11); a[0]=vx; a[1]=vy; a[2]=vw
    a[3:10]=np.clip(TGT-J(),-0.1,0.1); a[10]=grip
    return a
res={}
for k,v in LADDER.items():
    TGT=np.array(v)
    for i in range(22): step(act())
    e=J()-TGT; res[k]=float(np.abs(e).max())
    print(k,"maxerr",round(res[k],4),"err",np.round(e,3).tolist())
blocked=[k for k in LADDER if res[k]>0.03]
print("BLOCKED:",blocked)
choose = blocked[0] if blocked else "f30_d48"
# use one step deeper than first blocked for firm contact
keys=list(LADDER)
ci=min(keys.index(choose)+1,len(keys)-1) if blocked else len(keys)-1
choose=keys[ci]
print("CHOSEN",choose,LADDER[choose])
TGT=np.array(LADDER[choose])
for i in range(25): step(act())
print("held", np.round(J(),3).tolist(),"err",np.round(J()-TGT,3).tolist())
c0=cube(); print("cube0",np.round(c0,4).tolist(),"base",np.round(pose(),3).tolist())
def moveto(t,rate=0.1,maxn=45):
    for i in range(maxn):
        p=pose(); d=t-p[:2]
        if np.abs(d).max()<0.012: break
        step(act(vx=np.clip(d[0]*3,-rate,rate),vy=np.clip(d[1]*3,-rate,rate)))
yaw=pose()[2]; u=np.array([np.cos(yaw),np.sin(yaw)]); n=np.array([-u[1],u[0]])
print("yaw",round(yaw,3),"u",np.round(u,3).tolist(),"n",np.round(n,3).tolist())
def scan(sign,lat,nmax=28,rate=0.03):
    d=sign*u
    start=c0[:2]-d*0.50+n*lat
    moveto(start)
    print(" scan sign",sign,"lat",lat,"start",np.round(pose(),3).tolist())
    for i in range(nmax):
        step(act(vx=d[0]*rate,vy=d[1]*rate))
        c=cube()
        if np.abs(c[:2]-c0[:2]).max()>0.003:
            p=pose(); rel=c0[:2]-p[:2]
            print("  CONTACT base",np.round(p,4).tolist(),"cube",np.round(c,4).tolist(),
                  "fwd_off",round(float(rel@u),4),"lat_off",round(float(rel@n),4))
            return True
    print("  none; end base",np.round(pose(),3).tolist())
    return False
hit=False
for sign,lat in [(1,0.0),(1,0.09),(1,-0.09),(-1,0.0)]:
    if scan(sign,lat): hit=True; break
if hit:
    p0=pose().copy(); 
    for i in range(18):
        step(act(vx=u[0]*0.03*sign,vy=u[1]*0.03*sign))
        if i%3==0:
            p=pose(); c=cube()
            print("  push",i,"base",np.round(p[:2],4).tolist(),"cube",np.round(c,4).tolist(),
                  "rel_fwd",round(float((c[:2]-p[:2])@u),4),"rel_lat",round(float((c[:2]-p[:2])@n),4))
env.close()

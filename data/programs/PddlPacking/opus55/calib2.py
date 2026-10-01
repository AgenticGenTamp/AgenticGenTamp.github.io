from env_client import make_env
from pr2fk import *
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
T=env.observation_space.get_type
rob=obs.get_objects(T("robot"))[0]
blocks=obs.get_objects(T("block"))
def rstate(o):
    r=o.get_objects(T("robot"))[0]
    f=env.observation_space.type_features[T("robot")]
    return np.array([o.get(r,x) for x in f])
def bpose(o,b):
    return np.array([o.get(b,x) for x in ["pose_x","pose_y","pose_z","pose_qx","pose_qy","pose_qz","pose_qw","grasp_active"]])
def goto(o, base, q, grip=0.0):
    # step toward target until reached; returns obs, rejected flag
    for _ in range(40):
        s=rstate(o)
        d=np.zeros(11,dtype=np.float32)
        d[0:3]=np.array(base)-s[0:3]; d[3:10]=np.array(q)-s[3:10]
        d[2]=(d[2]+np.pi)%(2*np.pi)-np.pi
        if np.abs(d[:10]).max()<1e-4: return o,False
        d[:10]=np.clip(d[:10],-0.2,0.2); d[10]=grip
        o2,*_=env.step(d)
        if np.allclose(rstate(o2)[:10],s[:10]): print("REJ at",s[:10].round(3),"tgt",np.array(q).round(3)); return o2,True
        o=o2
    return o,False
b=blocks[1]; bp=bpose(obs,b); yaw=2*np.arctan2(bp[5],bp[6])
print("block",bp, yaw)
base=(-0.6,0.0,0.0)
s=rstate(obs); q=s[3:10]
Rdes=np.array([[0,np.sin(yaw)*0,0],[0,0,0],[0,0,0]])
# tool x=(0,0,-1), tool y = (cos yaw, sin yaw,0), z = x cross y
xa=np.array([0,0,-1.]); ya=np.array([np.cos(yaw),np.sin(yaw),0]); za=np.cross(xa,ya)
Rdes=np.stack([xa,ya,za],1)
obs,rej=goto(obs,base,q)
print("base moved", rstate(obs)[:3], rej)
for z in np.arange(1.15,0.6,-0.01):
    qn,err=ik(base,np.array([bp[0],bp[1],z]),Rdes,q)
    obs2,rej=goto(obs,base,qn)
    print(round(z,3),"err",round(err,4),"rej",rej, rstate(obs2)[3:10].round(3))
    if rej: break
    obs=obs2; q=qn
for g in [-1,0,0]:
    a=np.zeros(11,dtype=np.float32); a[10]=g
    obs,*_=env.step(a)
s=rstate(obs); print("grasp",s[10:], [bpose(obs,bb)[7] for bb in blocks])
print("blockpose", bpose(obs,b))
def tool_from_obs(o, b):
    s=rstate(o); bp=bpose(o,b)
    Rb=quat_to_R(bp[3:7]); tb=bp[:3]
    Rg=quat_to_R(s[15:19]); tg=s[12:15]
    Rt=Rb@Rg.T; tt=tb-Rt@tg
    return tt,Rt
rng=np.random.default_rng(0)
errs=[]
for i in range(30):
    s=rstate(obs)
    a=np.zeros(11,dtype=np.float32); a[3:10]=rng.uniform(-0.2,0.2,7); a[0:3]=rng.uniform(-0.1,0.1,3); a[1]=abs(a[1])-0.15; a[4]=-abs(a[4])
    obs2,*_=env.step(a); s2=rstate(obs2)
    if np.allclose(s2[:10],s[:10]): continue
    obs=obs2
    tt,Rt=tool_from_obs(obs,b)
    tf,Rf=fk(s2[:3],s2[3:10])
    errs.append((np.linalg.norm(tt-tf),np.abs(Rt-Rf).max()))
    if i<3: print(tt-tf)
print(np.array(errs).max(0), len(errs))

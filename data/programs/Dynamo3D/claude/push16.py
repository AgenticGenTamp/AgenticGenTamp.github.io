import numpy as np, sys
from env_client import make_env
seed=int(sys.argv[1]); ang=float(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed)
cn=sorted([n for n in obs.get_object_names() if n!='robot'])[int(sys.argv[3]) if len(sys.argv)>3 else 0]
def cp(o):
    c=o.get_object_from_name(cn); return np.array([float(o.get(c,'x')),float(o.get(c,'y'))])
def base(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,'pos_base_x')),float(o.get(r,'pos_base_y'))])
c0=cp(obs); th=np.radians(ang); dirv=np.array([np.cos(th),np.sin(th)])
stage=c0-dirv*1.5
# orbit staging: keep radius 1.5 from chair while rotating to the staging angle
b0=base(obs); v=b0-c0; r0=np.linalg.norm(v)
a0=np.arctan2(v[1],v[0]); a1=np.arctan2((stage-c0)[1],(stage-c0)[0])
da=(a1-a0+np.pi)%(2*np.pi)-np.pi
path=[]
K=max(3,int(abs(da)/0.15))
for i in range(1,K+1):
    ang_i=a0+da*i/K; rr=1.5
    path.append(c0+np.array([np.cos(ang_i),np.sin(ang_i)])*rr)
def goto(t,tol=0.06):
    global obs
    for _ in range(300):
        b=base(obs); d=t-b
        if np.linalg.norm(d)<tol: return True
        a=np.zeros(11,dtype=np.float32); a[:2]=np.clip(d,-0.1,0.1)
        obs,r,term,trunc,info=env.step(a)
        if term: return False
    return True
# first move out to radius 1.5 along current direction
if r0>1e-3:
    if not goto(c0+v/r0*1.5): print("term early"); sys.exit()
ok=True
for p in path:
    if not goto(p,0.1): ok=False; break
if not ok: print("term orbit"); sys.exit()
cs=cp(obs)
moved=np.linalg.norm(cs-c0)
path_len=0.0; prev=cp(obs)
for i in range(400):
    a=np.zeros(11,dtype=np.float32); a[:2]=dirv*0.04
    obs,r,term,trunc,info=env.step(a)
    cur=cp(obs); path_len+=float(np.linalg.norm(cur-prev)); prev=cur
    if abs(r+1)>1e-6: print("REW",round(r,4),np.round(cur,3),flush=True)
    if term:
        print("ang %g seed %d chair0 %s cstage %s cterm %s disp %.3f path %.3f bump %.3f robot %s"%(
            ang,seed,np.round(c0,3),np.round(cs,3),np.round(cur,3),float(np.linalg.norm(cur-c0)),path_len,moved,np.round(base(obs),3)))
        break
else:
    print("ang %g no term chair %s"%(ang,np.round(cp(obs),3)))

import numpy as np, sys
from env_client import make_env
seed=int(sys.argv[1]); ang=float(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed)
names=sorted([n for n in obs.get_object_names() if n!='robot'])
cn=names[0]
def cp(o):
    c=o.get_object_from_name(cn); return np.array([float(o.get(c,'x')),float(o.get(c,'y'))])
def base(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,'pos_base_x')),float(o.get(r,'pos_base_y'))])
c0=cp(obs); th=np.radians(ang); dirv=np.array([np.cos(th),np.sin(th)])
# stage behind chair
stage=c0-dirv*1.3
# go around: first move perpendicular away then to stage
perp=np.array([-dirv[1],dirv[0]])
wps=[base(obs)+ (stage-base(obs)) ]
def goto(t,avoid=True):
    global obs
    for _ in range(300):
        b=base(obs); d=t-b
        if np.linalg.norm(d)<0.06: return True
        step=np.clip(d,-0.1,0.1)
        nb=b+step*0.87
        if avoid and np.linalg.norm(nb-cp(obs))<0.9:
            # sidestep
            v=nb-cp(obs); v=v/np.linalg.norm(v)
            t2=cp(obs)+v*0.95+perp*0.3
            step=np.clip(t2-b,-0.1,0.1)
        a=np.zeros(11,dtype=np.float32); a[:2]=step
        obs,r,term,trunc,info=env.step(a)
        if abs(r+1)>1e-6: print("REWCHANGE",round(r,4),"chair",np.round(cp(obs),3),flush=True)
        if term: return False
    return True
ok=goto(stage)
res=[]
if ok:
    for i in range(200):
        b=base(obs)
        a=np.zeros(11,dtype=np.float32); a[:2]=dirv*0.03
        obs,r,term,trunc,info=env.step(a)
        res.append((round(r,4),np.round(cp(obs),3)))
        if abs(r+1)>1e-6: print("REWCHANGE",round(r,4),"chair",np.round(cp(obs),3),flush=True)
        if term:
            print("ang",ang,"TERM chair",np.round(cp(obs),3),"start",np.round(c0,3),"disp",round(float(np.linalg.norm(cp(obs)-c0)),3),flush=True); break
    else:
        print("ang",ang,"nopush end chair",np.round(cp(obs),3))
else:
    print("ang",ang,"term during staging, chair",np.round(cp(obs),3))

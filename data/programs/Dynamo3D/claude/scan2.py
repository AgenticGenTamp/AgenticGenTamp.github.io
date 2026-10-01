import sys, numpy as np
from env_client import make_env
seed=int(sys.argv[1]); ymin=float(sys.argv[2]); ymax=float(sys.argv[3]); xmin=float(sys.argv[4]); xmax=float(sys.argv[5])
env=make_env(); obs,info=env.reset(seed=seed)
names=[n for n in sorted(obs.get_object_names()) if n!='robot']
def base(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,f)) for f in ['pos_base_x','pos_base_y']])
def cps(o):
    return [np.array([float(o.get(o.get_object_from_name(n),'x')),float(o.get(o.get_object_from_name(n),'y'))]) for n in names]
CLR=0.85
def blocked(p):
    return any(np.linalg.norm(p-c)<CLR for c in cps(obs))
found=[]
def step_to(tgt):
    global obs
    for _ in range(400):
        b=base(obs); d=tgt-b
        if np.linalg.norm(d)<0.07: return 'ok'
        nb=b+np.clip(d,-0.1,0.1)*0.87
        if blocked(nb): return 'blocked'
        a=np.zeros(11,dtype=np.float32); a[:2]=np.clip(d,-0.1,0.1)
        obs,rew,term,trunc,info=env.step(a)
        if abs(rew+1.0)>1e-6:
            found.append((round(float(base(obs)[0]),3),round(float(base(obs)[1]),3),round(rew,4))); print("REW",found[-1],flush=True)
        if term: print("TERM",np.round(base(obs),3),flush=True); return 'term'
    return 'timeout'
y=ymin; i=0; nsteps=0
while y<=ymax+1e-9:
    xs=np.arange(xmin,xmax+1e-9,0.12)
    if i%2: xs=xs[::-1]
    for x in xs:
        p=np.array([x,y])
        if blocked(p): continue
        r=step_to(p)
        if r=='term': print("ABORT"); sys.exit()
    i+=1; y+=0.12
print("done",seed,ymin,ymax,"found",found)

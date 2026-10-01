import sys, numpy as np
from env_client import make_env
ymin,ymax,seed=float(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3])
env=make_env(); obs,info=env.reset(seed=seed)
def base(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,f)) for f in ['pos_base_x','pos_base_y']])
found=[]
xs=[-3.0,6.0]
def goto(tx,ty,maxit=300):
    global obs
    for _ in range(maxit):
        b=base(obs); d=np.array([tx,ty])-b
        if np.linalg.norm(d)<0.06: return True
        a=np.zeros(11,dtype=np.float32); a[0]=np.clip(d[0],-0.1,0.1); a[1]=np.clip(d[1],-0.1,0.1)
        obs,rew,term,trunc,info=env.step(a)
        if abs(rew+1.0)>1e-6:
            found.append((round(float(base(obs)[0]),3),round(float(base(obs)[1]),3),round(rew,4)))
            print("REW",found[-1],flush=True)
        if term:
            print("TERM",base(obs),rew,flush=True); return False
    return True
i=0
y=ymin
while y<=ymax+1e-9:
    tx=xs[i%2]
    if not goto(tx,y): break
    i+=1; y+=0.15
print("done band",ymin,ymax,"found",len(found))

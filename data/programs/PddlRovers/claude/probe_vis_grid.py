import numpy as np, sys
from probe_vis_lib import *
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 39
env=make_env(); obs,info=env.reset(seed=SEED, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
print("seed",SEED,"O",np.round(O,4))
pill=[(n,np.array([L[n]['x'],L[n]['y']])) for n in L if n.startswith('obstacle') and L[n]['half_z']>0.1]
mnd=[(n,np.array([L[n]['x'],L[n]['y']])) for n in L if n.startswith('obstacle') and L[n]['half_z']<0.1]
free,xs,res=build_grid(obs)
def vis(obs,i):
    if rf(obs,i)['calibrated']>0.5: obs,_,_,_,_=st(env,op='image',i=i)
    obs,_,_,_,_=st(env,op='calibrate',i=i)
    return obs, rf(obs,i)['calibrated']>0.5
ygrid=np.arange(0.2,2.01,0.2)
xgrid=np.arange(0.4,2.01,0.2)
rows=[]
recs=[]
for r,y in enumerate(ygrid[::-1]):
    line=""
    cols = xgrid if r%2==0 else xgrid[::-1]
    cells={}
    for x in cols:
        obs,ok=goto(env,obs,x,y,i=0,tol=0.02)
        p=pose(obs,0)[:2]
        if np.linalg.norm(p-np.array([x,y]))>0.05:
            obs,ok=nav(env,obs,x,y,i=0,tol=0.02,free=free,xs=xs,res=res)
            p=pose(obs,0)[:2]
        if np.linalg.norm(p-np.array([x,y]))>0.05:
            cells[x]='?'; continue
        obs,v=vis(obs,0)
        cells[x]='#' if v else '.'
        recs.append((float(p[0]),float(p[1]),int(v)))
    line="".join(cells[x] for x in xgrid)
    print("y=%+.1f  %s"%(y,line))
np.save('vis_grid_%d.npy'%SEED, np.array(recs))
print("x:", " ".join("%.1f"%x for x in xgrid))
print("pillars",[(n,tuple(np.round(p,3))) for n,p in pill])
print("mounds",[(n,tuple(np.round(p,3))) for n,p in mnd])
env.close()

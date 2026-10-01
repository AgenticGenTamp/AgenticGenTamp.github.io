from env_client import make_env
import numpy as np, sys
from tmodel import TGeom, sim_step, wrap, corners_world
env = make_env()
rng=np.random.default_rng(0)
rows=[]
for seed in range(4):
    obs,_=env.reset(seed=seed)
    g=TGeom(*obs[12:15])
    for k in range(150):
        tgt = obs[0:2] + rng.normal(0,0.4,2)
        d=tgt-obs[16:18]; a=np.clip(d/max(np.linalg.norm(d),1e-9)*0.0499,-0.0499,0.0499)
        A=lambda v: np.array([v])
        bx,by,bt,rx,ry=sim_step(g,A(obs[0]),A(obs[1]),A(obs[2]),A(obs[16]),A(obs[17]),a[0],a[1])
        obs2,*_=env.step(a)
        moved=np.abs(obs2[0:3]-obs[0:3]).max()
        e=np.r_[obs2[0]-bx[0],obs2[1]-by[0],wrap(obs2[2]-bt[0])]
        cx,cy=corners_world(g,obs2[0],obs2[1],obs2[2]); inside=(cx.min()>0.15)&(cx.max()<4.85)&(cy.min()>0.15)&(cy.max()<4.85)
        c_,s_=np.cos(obs[2]),np.sin(obs[2]); lx=c_*(obs[16]-obs[0])+s_*(obs[17]-obs[1]); ly=-s_*(obs[16]-obs[0])+c_*(obs[17]-obs[1]); px,py,nx,ny,dd=g.closest(lx,ly); lu=np.array([c_*a[0]+s_*a[1],-s_*a[0]+c_*a[1]]); un=-(lu[0]*nx+lu[1]*ny)
        if inside and moved>1e-7: rows.append((dd-0.1,un,np.abs(e).max(),moved))
        obs=obs2
R=np.array(rows)
print('n',len(R),'mean err',R[:,2].mean(),'p95',np.percentile(R[:,2],95),'max',R[:,2].max())
for r in R[np.argsort(-R[:,2])][:8]: print(np.round(r,4))

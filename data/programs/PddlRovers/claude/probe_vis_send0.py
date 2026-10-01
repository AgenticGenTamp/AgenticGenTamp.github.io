import numpy as np
from probe_vis_lib import *
env=make_env()
pick=None
for seed in range(60):
    obs,info=env.reset(seed=seed, options={'object_count':1})
    L=layout(obs)
    es=[n for n in L if n.startswith('sample') and 0.5<L[n]['x']<1.9 and abs(L[n]['y'])<1.7]
    if es: pick=(seed,es[0],np.array([L[es[0]]['x'],L[es[0]]['y']])); break
env.close()
seed,s,sp=pick; print("seed",seed,s,np.round(sp,2))
env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':1})
L=layout(obs); free,xs,res=build_grid(obs); lan=np.array([L['lander']['x'],L['lander']['y']])
got=False
for ang in np.arange(0,2*np.pi,np.pi/6):
    Q=sp+0.16*np.array([np.cos(ang),np.sin(ang)])
    if abs(Q[0])<0.32 or abs(Q[0])>2.05 or abs(Q[1])>2.05: continue
    obs,ok=nav(env,obs,Q[0],Q[1],i=0,tol=0.03,free=free,xs=xs,res=res)
    if np.linalg.norm(pose(obs,0)[:2]-sp)<0.245:
        obs,_,_,_,_=st(env,op='sample',i=0)
        if feats(obs,s)['analyzed_rover0']>0.5: got=True; break
print("analyzed_rover0=%s"%got)
if got:
    for P in [(0.33,-2.10),(0.33,-1.50),(0.60,-2.10),(1.20,-2.10)]:
        obs,ok=nav(env,obs,P[0],P[1],i=0,tol=0.05,free=free,xs=xs,res=res)
        p=pose(obs,0)[:2]; d=np.linalg.norm(p-lan)
        obs,_,_,_,_=st(env,op='send',i=0)
        print("  rover0 send at %s lander_dist=%.2f -> received=%.0f"%(np.round(p,2),d,feats(obs,s)['received_analysis']))
        if feats(obs,s)['received_analysis']>0.5: break
env.close()

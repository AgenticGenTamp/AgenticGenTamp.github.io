import numpy as np
from probe_vis_lib import *
env=make_env(); obs,info=env.reset(seed=1, options={'object_count':1})
L=layout(obs); free,xs,res=build_grid(obs); lan=np.array([L['lander']['x'],L['lander']['y']])
sp=np.array([L['sample4']['x'],L['sample4']['y']]); print("sample4",np.round(sp,2),"lander",lan)
got=False
for ang in np.arange(0,2*np.pi,np.pi/6):
    Q=sp+0.16*np.array([np.cos(ang),np.sin(ang)])
    if abs(Q[0])<0.32 or abs(Q[0])>2.05 or abs(Q[1])>2.05: continue
    obs,ok=nav(env,obs,Q[0],Q[1],i=0,tol=0.03,free=free,xs=xs,res=res)
    if np.linalg.norm(pose(obs,0)[:2]-sp)<0.245:
        obs,_,_,_,_=st(env,op='sample',i=0)
        if feats(obs,'sample4')['analyzed_rover0']>0.5: got=True; break
print("analyzed:",got,"pos",np.round(pose(obs,0)[:2],2),"lander dist %.2f"%np.linalg.norm(pose(obs,0)[:2]-lan))
p0=pose(obs,0)[:2].copy(); u=(lan-p0)/np.linalg.norm(lan-p0)
d=np.linalg.norm(p0-lan); prev=d
for k in range(40):
    obs,_,_,_,_=st(env,op='send',i=0)
    if feats(obs,'sample4')['received_analysis']>0.5:
        print("SEND threshold: first success at lander-dist=%.3f (prev fail %.3f) pos=%s"%(d,prev,np.round(pose(obs,0)[:2],2))); break
    prev=d
    tgt=lan+u*-1*0  # dummy
    tgt=lan + (p0-lan)/np.linalg.norm(p0-lan)*(d-0.05)
    obs,_=goto(env,obs,tgt[0],tgt[1],i=0,tol=0.01)
    nd=np.linalg.norm(pose(obs,0)[:2]-lan)
    if abs(nd-d)<1e-3: print("blocked while approaching at d=%.2f"%nd); break
    d=nd
else: print("no success, d=%.2f"%d)
env.close()

import numpy as np
from probe_vis_lib import *
# A) rover1 west-side, pillar dead-on at increasing frac (seed 22)
env=make_env(); obs,info=env.reset(seed=22, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
q=np.array([L['obstacle10']['x'],L['obstacle10']['y']])
print("seed22 obstacle10 feats",feats(obs,'obstacle10'))
print("objective feats",feats(obs,'objective0'))
free,xs,res=build_grid(obs)
A=-0.085
dqo=np.linalg.norm(q-O); u=(q-O)/dqo
def camof(obs,i):
    p=pose(obs,i); return p[:2]+A*np.array([np.cos(p[2]),np.sin(p[2])])
first=True
for Lb in [0.25,0.5,0.75,1.0,1.15]:
    # want camera collinear: place center so that camera lands on the line
    P=q+u*Lb
    p_t=P - A*np.array([np.cos(0.0),np.sin(0.0)])  # theta=0 assumed
    if first:
        obs,ok=nav(env,obs,p_t[0],p_t[1],i=1,tol=0.01,free=free,xs=xs,res=res); first=False
    else:
        obs,ok=goto(env,obs,p_t[0],p_t[1],i=1,tol=0.005)
    p=pose(obs,1)
    if np.linalg.norm(p[:2]-p_t)>0.03: print("  L=%.2f unreachable (at %s)"%(Lb,np.round(p[:2],2))); continue
    c=camof(obs,1); d=np.linalg.norm(O-c); uu=(O-c)/d; w=q-c
    perp=abs(w[0]*uu[1]-w[1]*uu[0]); frac=np.dot(w,uu)/d
    obs,v=vis_test(env,obs,1)
    print("  rover1 L=%.2f d=%.3f perp=%.4f frac=%.3f vis=%s theta=%.2f"%(Lb,d,perp,frac,v,p[2]))
env.close()
# B) rover0 east-side seed 39: pillar9 dead-on at several fracs
env=make_env(); obs,info=env.reset(seed=39, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
q=np.array([L['obstacle9']['x'],L['obstacle9']['y']])
free,xs,res=build_grid(obs)
dqo=np.linalg.norm(q-O); u=(q-O)/dqo
print("seed39 pillar9 dist to O",dqo)
first=True
for Lb in [0.3,0.5,0.8,1.0,1.2]:
    P=q+u*Lb
    p_t=P - A*np.array([np.cos(np.pi),np.sin(np.pi)])
    if first:
        obs,ok=nav(env,obs,p_t[0],p_t[1],i=0,tol=0.01,free=free,xs=xs,res=res); first=False
    else:
        obs,ok=goto(env,obs,p_t[0],p_t[1],i=0,tol=0.005)
    p=pose(obs,0)
    if np.linalg.norm(p[:2]-p_t)>0.03: print("  L=%.2f unreachable (at %s)"%(Lb,np.round(p[:2],2))); continue
    c=p[:2]+A*np.array([np.cos(p[2]),np.sin(p[2])]); d=np.linalg.norm(O-c); uu=(O-c)/d; w=q-c
    perp=abs(w[0]*uu[1]-w[1]*uu[0]); frac=np.dot(w,uu)/d
    obs,v=vis_test(env,obs,0)
    print("  rover0 L=%.2f d=%.3f perp=%.4f frac=%.3f vis=%s theta=%.2f"%(Lb,d,perp,frac,v,p[2]))
env.close()

import numpy as np, sys
from probe_vis_lib import *
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 13
PIL=sys.argv[2] if len(sys.argv)>2 else 'obstacle7'
env=make_env(); obs,info=env.reset(seed=SEED, options={'object_count':1})
L=layout(obs)
O=np.array([L['objective0']['x'],L['objective0']['y']])
Q=np.array([L[PIL]['x'],L[PIL]['y']])
u=(Q-O)/np.linalg.norm(Q-O); nrm=np.array([-u[1],u[0]])
print("O",O,"Q",Q,"|QO|",np.linalg.norm(Q-O))
free,xs,res=build_grid(obs)
def seg_pt_dist(a,b,p):
    ab=b-a; t=np.clip(np.dot(p-a,ab)/np.dot(ab,ab),0,1); return np.linalg.norm(a+t*ab-p)
def vis(obs):
    if rf(obs,0)['calibrated']>0.5: obs,_,_,_,_=st(env,op='image',i=0)
    obs,_,_,_,_=st(env,op='calibrate',i=0)
    return obs, rf(obs,0)['calibrated']>0.5
def at(obs,P,first=False):
    if first: obs,ok=nav(env,obs,P[0],P[1],i=0,tol=0.003,free=free,xs=xs,res=res)
    else: obs,ok=goto(env,obs,P[0],P[1],i=0,tol=0.003)
    return obs,ok
first=True
for Lb in [0.5, 1.0]:
    P0=Q+u*Lb
    if abs(P0[0])<0.45 or abs(P0[0])>2.05 or abs(P0[1])>2.05: print("skip L",Lb); continue
    print("--- L=%.2f base %s dist_to_O=%.3f"%(Lb,np.round(P0,3),np.linalg.norm(P0-O)))
    rows=[]
    for s in np.arange(-0.30,0.301,0.05):
        P=P0+nrm*s
        obs,ok=at(obs,P,first); first=False
        p=pose(obs,0)[:2]
        obs,v=vis(obs)
        pd=seg_pt_dist(p,O,Q)
        rows.append((round(float(s),3),round(float(pd),4),v,round(float(np.linalg.norm(p-O)),3),ok))
    for r in rows: print("  s=%+.3f perp=%.4f vis=%s d=%.3f ok=%s"%r)
env.close()

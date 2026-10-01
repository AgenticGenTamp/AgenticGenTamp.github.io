import numpy as np
from probe_vis_lib import *
env=make_env()
pick=None
for seed in range(100):
    obs,info=env.reset(seed=seed, options={'object_count':1})
    L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
    if O[0]>-0.6: continue
    stones=[n for n in L if n.startswith('sample') and L[n]['is_soil']<0.5 and -2.0<L[n]['x']<-0.5 and abs(L[n]['y'])<1.9]
    soils=[n for n in L if n.startswith('sample') and L[n]['is_soil']>0.5 and -2.0<L[n]['x']<-0.5 and abs(L[n]['y'])<1.9]
    if stones and soils:
        pick=(seed,O,stones[0],soils[0],np.array([L[stones[0]]['x'],L[stones[0]]['y']]),np.array([L[soils[0]]['x'],L[soils[0]]['y']]))
        break
env.close()
seed,O,stn,sol,sp_stn,sp_sol=pick
print("using seed",seed,"O",np.round(O,3),stn,np.round(sp_stn,2),sol,np.round(sp_sol,2))
env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':1})
L=layout(obs); lan=np.array([L['lander']['x'],L['lander']['y']])
free,xs,res=build_grid(obs)
nsteps=[0]; treward=[0.0]
_st=st
def S(**kw):
    o,r,te,tr,inf=_st(env,**kw); nsteps[0]+=1; treward[0]+=r
    return o,r,te,tr,inf
import probe_vis_lib as lib
lib.st=lambda e,**kw: S(**kw)
def approach(obs,target,i=1,off=0.18):
    for ang in np.arange(0,2*np.pi,np.pi/4):
        P=target+off*np.array([np.cos(ang),np.sin(ang)])
        if abs(P[0])<0.30 or abs(P[0])>2.1 or abs(P[1])>2.1: continue
        obs,ok=lib.nav(env,obs,P[0],P[1],i=i,tol=0.03,free=free,xs=xs,res=res)
        if np.linalg.norm(lib.pose(obs,i)[:2]-target)<0.249: return obs,True
    return obs,False
# 1) stone
obs,ok=approach(obs,sp_stn); obs,_,_,_,_=S(op='sample',i=1)
print("stone sampled analyzed=%.0f store=%.0f (steps %d)"%(feats(obs,stn)['analyzed_rover1'],rf(obs,1)['store_full'],nsteps[0]))
obs,_,_,_,_=S(op='drop',i=1)
# 2) soil
obs,ok=approach(obs,sp_sol); obs,_,_,_,_=S(op='sample',i=1)
print("soil sampled analyzed=%.0f dist=%.3f (steps %d)"%(feats(obs,sol)["analyzed_rover1"],np.linalg.norm(lib.pose(obs,1)[:2]-sp_sol),nsteps[0]))
obs,_,_,_,_=S(op='drop',i=1)
# 3) image objective: try positions at 1.2m on an arc
done=False
for ang in np.arange(-np.pi,0.01,np.pi/8):
    P=O+1.2*np.array([np.cos(ang),np.sin(ang)])
    if abs(P[0])<0.35 or abs(P[0])>2.05 or abs(P[1])>2.05: continue
    obs,ok=lib.nav(env,obs,P[0],P[1],i=1,tol=0.03,free=free,xs=xs,res=res)
    if np.linalg.norm(lib.pose(obs,1)[:2]-P)>0.08: continue
    obs,_,_,_,_=S(op='calibrate',i=1)
    if rf(obs,1)['calibrated']>0.5:
        obs,_,_,_,_=S(op='image',i=1)
        if feats(obs,'objective0')['have_image_rover1']>0.5: done=True; break
print("image acquired=%s (steps %d)"%(done,nsteps[0]))
# 4) send near lander
obs,ok=lib.nav(env,obs,lan[0]+1.0,lan[1]+0.9,i=1,tol=0.05,free=free,xs=xs,res=res)
obs,r,te,tr,inf=S(op='send',i=1)
print("after send: recv_stone=%.0f recv_soil=%.0f recv_image=%.0f term=%s (steps %d)"%(
  feats(obs,stn)['received_analysis'],feats(obs,sol)['received_analysis'],feats(obs,'objective0')['received_image'],te,nsteps[0]))
# 5) go home
obs,ok=lib.nav(env,obs,-1.0,-1.75,i=1,tol=0.01,free=free,xs=xs,res=res)
print("home nav ok=%s pos=%s theta=%.2f at_home=%.0f"%(ok,np.round(lib.pose(obs,1)[:2],3),lib.pose(obs,1)[2],rf(obs,1)['at_home']))
# fix theta to 0 (initial for rover1)
for _ in range(30):
    d=(0.0-lib.pose(obs,1)[2]+np.pi)%(2*np.pi)-np.pi
    if abs(d)<0.003: break
    obs,r,te,tr,inf=S(dth=float(np.clip(d,-0.4,0.4)),i=1)
obs,r,te,tr,inf=S(op='noop',i=1)
print("FINAL: r0 at_home=%.0f store=%.0f | r1 at_home=%.0f store=%.0f theta=%.3f"%(
  rf(obs,0)['at_home'],rf(obs,0)['store_full'],rf(obs,1)['at_home'],rf(obs,1)['store_full'],lib.pose(obs,1)[2]))
print("terminated=%s truncated=%s reward_last=%s total_reward=%.1f steps=%d info=%s"%(te,tr,r,treward[0],nsteps[0],inf))
env.close()

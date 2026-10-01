import math,sys
import numpy as np
from scipy.optimize import least_squares
from probe_grasp import fk
from search_grasp_grid_local import setup,vals,g,move

LO=np.array([-.715,-.524,-.8,-2.321,-math.pi,-2.094,-math.pi])
HI=np.array([2.285,1.396,3.9,0,math.pi,0,math.pi])
seed=int(sys.argv[1]) if len(sys.argv)>1 else 101
e,s,p=setup(seed);base=vals(s,p)[:3];e.close();R=fk(base,p.Q)[:3,:3];pos=fk(base,p.Q)[:3,3]
rng=np.random.default_rng(77+seed);cands=[]
for q2 in (.03,.05,.08,.10,.12,.15):
 def fun(x):
  q=np.insert(x,1,q2);T=fk(base,q);D=T[:3,:3].T@R
  rv=np.array([D[2,1]-D[1,2],D[0,2]-D[2,0],D[1,0]-D[0,1]])/2
  return np.r_[20*(T[:3,3]-pos),3*rv]
 lo=np.delete(LO,1);hi=np.delete(HI,1)
 for _ in range(50):
  z=least_squares(fun,rng.uniform(lo,hi),bounds=(lo,hi),max_nfev=500)
  q=np.insert(z.x,1,q2)
  if np.linalg.norm(fun(z.x))<.015 and min([np.linalg.norm(q-x) for x in cands] or [9])>.15:cands.append(q)
cands=sorted(cands,key=lambda q:np.linalg.norm(q-p.Q))[:50]
for k,q in enumerate(cands):
 e,s,p=setup(seed);v=vals(s,p);target=np.r_[v[:3],q]
 # interpolate slowly, closing throughout; record actual accepted endpoint
 s,ok=move(e,s,p,target,True,50,.04);actual=vals(s,p)
 if ok:
  print('HIT',seed,k,'q',q.tolist(),'actual',actual.tolist(),flush=True);e.close();break
 if k%5==0: print('try',k,'q',np.round(q,3).tolist(),'actual',np.round(actual[3:],3).tolist(),flush=True)
 e.close()
else:print('NONE',seed,len(cands))

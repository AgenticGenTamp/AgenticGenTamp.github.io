from env_client import make_env
from grasp_baseline import GeneratedApproach
from kinematics_candidate import forward
from scipy.optimize import least_squares
import numpy as np,sys

def pik(z,mount):
 def q(v):return np.array([0,v[0],np.pi,v[1],0,v[0]-v[1]-np.pi,np.pi/2])
 def f(v):return (forward(q(v),.12,(0,0,mount))[:3,3]-[.55,.001,z])[[0,2]]
 v=least_squares(f,[1.5,-1.2],bounds=([-2.2,-2.58],[2.2,2.58]),max_nfev=50).x;return q(v)
if __name__=='__main__':
 for m in [.36,.38,.4]:
  for dx in [-.10,-.12,-.14]:
   e=make_env();s,i=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);p.qs=[pik(z,m) for z in [.15,.018,.22]];p.b[0]+=dx;mx=0
   for k in range(310):
    s,*_=e.step(p.get_action(s));mx=max(mx,s[2])
    if p.phase==3 and p.age>60:break
   print('PLANAR',m,dx,mx,s[:3],flush=True);e.close()

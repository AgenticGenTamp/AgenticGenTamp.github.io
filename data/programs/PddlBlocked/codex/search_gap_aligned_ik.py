"""Solve clipped-base IK while preserving the pen-gap grasp frame."""
import math, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from env_client import make_env
from approach import GeneratedApproach
from probe_grasp import fk
from search_grasp_grid_local import setup, vals, g, move

LO=np.array([-.715,-.524,-.8,-2.321,-math.pi,-2.094,-math.pi])
HI=np.array([2.285,1.396,3.9,0,math.pi,0,math.pi])

def solve(seed):
 e,s,p=setup(seed)
 green=np.array([g(s,'green0','pose_'+x) for x in 'xyz'])
 # Restore gap-aligned heading from the initial blocker->green axis.
 theta=p.w(math.atan2(-p.out[1],-p.out[0])-p.TOOL_YAW)
 c,z=math.cos(theta),math.sin(theta)
 off=np.array([c*p.OFF[0]-z*p.OFF[1],z*p.OFF[0]+c*p.OFF[1]])
 ideal=np.r_[green[:2]-off,theta]
 # Keep the collision-free north-east tangent base location, but restore the
 # gap-aligned heading and compensate with arm redundancy.
 base=np.array([5., .92, 1.425])
 Tg=fk(ideal,p.Q); Rg=Tg[:3,:3]
 rng=np.random.default_rng(seed+44000);sol=[]
 for k in range(160):
  def fun(q):
   T=fk(base,q)
   # Only the approach axis must follow the narrow gap; wrist roll is free.
   return np.r_[25*(T[:3,3]-Tg[:3,3]),
                3*np.cross(T[:3,0],Rg[:,0]),.003*(q-p.Q)]
  q=least_squares(fun,rng.uniform(LO,HI),bounds=(LO,HI),max_nfev=800).x
  if np.linalg.norm(fk(base,q)[:3,3]-Tg[:3,3])<.01 and np.linalg.norm(np.cross(fk(base,q)[:3,0],Rg[:,0]))<.08 and min([np.linalg.norm(q-x) for x in sol]or[9])>.1:sol.append(q)
 print('geom','ideal',ideal,'base',base,'overrun',ideal[0]-5,'sols',len(sol),flush=True)
 e.close()
 for k,q in enumerate(sorted(sol,key=lambda x:np.linalg.norm(x-p.Q))):
  e,s,p=setup(seed);cur=vals(s,p)
  # retreat north, establish new heading and lifted redundant pose, then enter.
  high=q.copy();high[1]=max(LO[1],q[1]-.20)
  first=cur.copy();first[:2]=[5.,1.60]
  s,_=move(e,s,p,first,False,40,.04)
  for xy in (np.array([5.,1.60]),base[:2]):
   target=np.r_[xy,base[2],high]
   s,_=move(e,s,p,target,False,80,.04)
  s,ok=move(e,s,p,np.r_[base,q],True,60,.018)
  actual=vals(s,p)
  if ok:
   print('HIT',seed,k,'theta',theta,'base',base.tolist(),'q',q.tolist(),'actual',actual.tolist(),flush=True);e.close();return
  if k%5==0:print('try',k,'q',np.round(q,3),'actual',np.round(actual,3),flush=True)
  e.close()
 print('NONE',seed)

if __name__=='__main__':solve(int(sys.argv[1]) if len(sys.argv)>1 else 101)

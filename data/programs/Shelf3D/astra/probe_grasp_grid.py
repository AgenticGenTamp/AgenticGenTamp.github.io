from env_client import make_env
from kinematics import *
from concurrent.futures import ThreadPoolExecutor
import sys

def run(args):
 z,dx,dy,grip=args
 E=make_env();s,info=E.reset(seed=42);r=s.get_object_from_name('robot');o=s.get_object_from_name('cube1');xyz=np.array([s.get(o,f) for f in ['x','y','z']]);base=np.r_[xyz[:2]-[dx,dy],0]
 def pose(h):
  def res(x):
   q=HOME.copy();q[[1,3,5]]=x;p,R=fk(q)
   return np.r_[(p-np.array([.5,.001,h]))[[0,2]],R[0,2]]
  q=HOME.copy();q[[1,3,5]]=least_squares(res,HOME[[1,3,5]],max_nfev=100).x
  return q
 down=pose(z);up=pose(.3)
 maxz=0
 for phase,(q,g,n) in enumerate([(HOME,1-grip,35),(down,1-grip,110),(down,grip,15),(up,grip,80)]):
  for i in range(n):
   cur=np.array([s.get(r,'pos_arm_joint'+str(j)) for j in range(1,8)]);b=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
   a=np.r_[np.clip(base-b,-.1,.1),np.clip((q-cur)*2,-.1,.1),g].astype(np.float32)
   s,rew,t,tr,info=E.step(a)
   maxz=max(maxz,s.get(o,'z'))
  if phase>=1:print(args,'phase',phase,'xyz',np.round([s.get(o,f) for f in ['x','y','z']],4),'qerr',np.round(np.max(np.abs(q-cur)),3),flush=True)
 E.close();return maxz
if __name__=='__main__':
 jobs=[(z,.65,0,g) for z in [-.06,-.02,0] for g in [0,1]]
 with ThreadPoolExecutor(3) as ex:print('results',list(zip(jobs,ex.map(run,jobs))))

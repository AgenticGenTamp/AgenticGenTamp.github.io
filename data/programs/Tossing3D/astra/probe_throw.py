import numpy as np
from env_client import make_env
from approach import GeneratedApproach
from kinova_candidate import fk
import sys

def run(seed=0,kick=1.):
 e=make_env();s,info=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
 rob=p.robot;c=p.cube;b=s.get_object_from_name('bin_0')
 def xyz(o):return np.array([s.get(o,f) for f in ['x','y','z']])
 def qnow():return np.array([s.get(rob,'pos_arm_joint'+str(j)) for j in range(1,8)])
 def control(q,base=None,g=1):
  a=np.zeros(18);err=q-qnow();a[3:10]=np.clip(err,-.1,.1);a[11:]=5*err;a[10]=g
  if base is not None:a[:3]=np.clip(base-np.array([s.get(rob,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']]),-.1,.1)
  return a
 for k in range(125):s,*_=e.step(p.get_action(s))
 qt=p.ik(.4,.85);base=np.array([.9,xyz(b)[1]-.00135,0])
 for k in range(55):s,*_=e.step(control(qt,base))
 print('READY',seed,xyz(c),xyz(b),'q',qnow(),flush=True)
 qend=p.ik(.85,1.0)
 for k in range(16):
  if k==0:
   a=control(qend);a[11:]=kick*5*(qend-qnow())
  else:a=control(qend,g=0)
  s,r,t,tr,i=e.step(a)
  print(k,'p',xyz(c).round(4),'v',np.array([s.get(c,f) for f in ['vx','vy','vz']]).round(4),'r',r,t,flush=True)
 e.close()
if __name__=='__main__':run(int(sys.argv[1]) if len(sys.argv)>1 else 0,float(sys.argv[2]) if len(sys.argv)>2 else 1.)

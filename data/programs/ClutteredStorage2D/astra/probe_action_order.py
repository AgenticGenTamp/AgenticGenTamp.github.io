import numpy as np
from env_client import make_env
from approach import GeneratedApproach

def run(kind):
 e=make_env();s,info=e.reset(seed=54 if kind=='pregrasp' else 0);a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,info)
 def block():
  b=s.get_object_from_name(a.target)
  return np.array([float(s.get(b,f)) for f in ['x','y','theta']])
 for i in range(1000):
  oldphase=a.phase
  ac=a.get_action(s)
  if kind=='release' and oldphase=='release' and a.phase=='retreat':
   before=block();ac[3]=-.1
   s,*_=e.step(ac)
   print('RELEASE',i,'before',before,'after',block(),'delta',block()-before,flush=True);break
  if kind=='pregrasp' and a.phase=='grasp' and a.path:
   goal=a.path[-1]
   nextq=a.q+ac[:4]
   d=goal-nextq;d[2]=(d[2]+np.pi)%(2*np.pi)-np.pi
   if len(a.path)==1 and np.max(abs(d))<.0002:
    before=block();ac[4]=1;s,*_=e.step(ac)
    atcontact=block()
    s,*_=e.step(np.array([0,0,0,.002,1],np.float32))
    print('PREGRASP',i,'before',before,'contact',atcontact,'aftertest',block(),'testdelta',block()-atcontact,flush=True);break
  s,r,done,trunc,inf=e.step(ac)
  if done or trunc:
   print('EARLY_END',kind,i,done,trunc,flush=True);break
 e.close()
for kind in ['pregrasp','release']:run(kind)

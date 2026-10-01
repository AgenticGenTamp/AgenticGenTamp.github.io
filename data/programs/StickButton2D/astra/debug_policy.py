from env_client import make_env
from approach import GeneratedApproach
import sys,numpy as np
E=make_env();s,info=E.reset(seed=int(sys.argv[1]),options={'object_count':int(sys.argv[2])} if len(sys.argv)>2 else None)
p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,info)
prev=None
for k in range(1000):
 a=p.get_action(s);s,r,done,trunc,i=E.step(a)
 robot=s.get_object_from_name('robot');stick=s.get_object_from_name('stick')
 rem=tuple(sorted(n for n in s.get_object_names() if n.startswith('button') and s.get(s.get_object_from_name(n),'color_g')<.5))
 key=(p.phase,rem)
 if key!=prev or k%50==0:
  print(k,p.phase,'robot',[round(s.get(robot,f),4) for f in ['x','y','theta']],'stick',[round(s.get(stick,f),4) for f in ['x','y','theta']],'action',np.round(a,4),'goal',p.goal,'rem',rem,flush=True)
 prev=key
 if done or trunc:break
E.close()

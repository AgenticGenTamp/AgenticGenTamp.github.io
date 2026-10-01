from env_client import make_env
from policy_variants import GeneratedApproach
import sys,time
for seed in map(int,sys.argv[1:] or range(4)):
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
 start=time.time()
 for step in range(e.max_steps):
  a=p.get_action(s);s,rew,term,trunc,info=e.step(a)
  if term or trunc:break
 pts=[(round(s.get(o,'x'),3),round(s.get(o,'y'),3)) for t in ('small_circle','small_square') for o in s.get_objects(e.observation_space.get_type(t))]
 print(seed,'success',term,'steps',step+1,'seconds',round(time.time()-start,1),'cycle',p.cycle,'phase',p.phase,'count',len(pts),'positions',pts,flush=True)
 e.close()

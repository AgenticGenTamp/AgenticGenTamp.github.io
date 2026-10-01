from env_client import make_env
from approach import GeneratedApproach
for count in [5,6]:
 e=make_env();s,i=e.reset(seed=34,options={'object_count':count});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
 print('COUNT',count)
 for step in range(20):
  pos=[float(s.get(p.robot,f)) for f in ['x','y','theta']]
  act=p.get_action(s);print(step,'pos',pos,'act',act,'path',[x.tolist() for x in p.path[:4]],'stuck',p.stuck,'smooth',p.smoothing)
  s,r,d,t,i=e.step(act)
 e.close()

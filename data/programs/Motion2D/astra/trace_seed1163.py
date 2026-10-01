from env_client import make_env
from approach import GeneratedApproach

e=make_env();s,i=e.reset(seed=1163);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
for n in s.get_object_names():
 o=s.get_object_from_name(n);fs=['x','y','theta'] if n=='robot' else ['x','y','width','height'];print(n,[float(s.get(o,f)) for f in fs])
for step in range(20):
 pos=[float(s.get(p.robot,f)) for f in ['x','y','theta']]
 act=p.get_action(s);print(step,'pos',pos,'act',act.tolist(),'path',[x.tolist() for x in p.path[:3]],'stuck',p.stuck,'smooth',p.smoothing)
 s,r,d,t,i=e.step(act)
e.close()

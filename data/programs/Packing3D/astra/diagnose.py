import sys,numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]);count=int(sys.argv[2]) if len(sys.argv)>2 else None
e=make_env();s,info=e.reset(seed=seed,options={'object_count':count} if count else None);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
print('LAYOUT',p.layout,flush=True)
for t in e.observation_space.types:
 for o in s.get_objects(t):
  if o.name.startswith('part'):print(o.name,o.type.name,{f:s.get(o,f) for f in e.observation_space.type_features[t]},flush=True)
for i in range(65):
 a=p.get_action(s);ns,r,te,tr,inf=e.step(a)
 print(i,p.phase,p.target.name if p.target else '',np.round(a,4),'held',[(n,ns.get(ns.get_object_from_name(n),'grasp_active')) for n in p.parts], 'q',p.config(ns)[[0,1,3,4,8]],flush=True)
 s=ns
 if te:print('SUCCESS',flush=True);break
e.close()

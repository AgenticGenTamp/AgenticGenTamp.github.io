from env_client import make_env
from approach import GeneratedApproach
import sys
seed=int(sys.argv[1]);e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
for n in s.get_object_names():
 o=s.get_object_from_name(n); print('initial',n,[(f,s.get(o,f)) for f in ['x','y','theta']])
for k in range(150):
 a=p.get_action(s)
 if k<10 or k%10==0:print(k,p.phase,'a',a,'r',p.prev,'pick',getattr(p,'pick_goal',None),'stick',[(f,s.get(s.get_object_from_name('stick'),f)) for f in ['x','y','theta']])
 s,r,t,tr,i=e.step(a)
 if t:print('DONE',k);break
e.close()

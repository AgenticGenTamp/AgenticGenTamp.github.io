from env_client import make_env
from approach import GeneratedApproach
import sys
seed=int(sys.argv[1]);e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
for k in range(180):
 a=p.get_action(s);r=p.robot;b=p.item
 print(k,p.phase,'item',b.name,'r',*[round(s.get(r,f),4) for f in ['x','y']], 'b',*[round(s.get(b,f),4) for f in ['x','y','height']], 'a',a.round(4).tolist(),flush=True)
 s,r,te,tr,i=e.step(a)
 if te:print('SUCCESS');break

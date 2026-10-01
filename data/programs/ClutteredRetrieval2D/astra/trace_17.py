from env_client import make_env
from approach import GeneratedApproach
import time
E=make_env();s,i=E.reset(seed=17);p=GeneratedApproach(E.action_space,E.observation_space,{})
p.reset(s,i);start=time.time();last=None
for k in range(300):
 a=p.get_action(s);s,r,d,tr,i=E.step(a)
 phase=(p.phase,p.chosen)
 if phase!=last or k%20==0:print(k,phase,'q',p.q.round(4),'a',a.round(4),'held',p.held,'stuck',p.stuck,'fails',p.failures,'time',round(time.time()-start,2),flush=True)
 last=phase
 if d or tr:print('END',d,tr,k);break
E.close()

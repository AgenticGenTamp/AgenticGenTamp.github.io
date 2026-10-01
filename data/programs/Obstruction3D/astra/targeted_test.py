from env_client import make_env
from approach import GeneratedApproach
import sys,time
seed=int(sys.argv[1]);count=int(sys.argv[2]);e=make_env();s,i=e.reset(seed=seed,options={'object_count':count});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);start=time.time();old=None
for t in range(1000):
 s,r,done,tr,info=e.step(p.get_action(s));key=(p.index,p.stage,p.retry)
 if key!=old or t%50==0:
  print(t,key,'obj',p.obj.name,'pos',p.pos(s,p.obj).round(5),'stuck',p.stuck,flush=True)
 old=key
 if done:break
print('RESULT',seed,count,done,t+1,time.time()-start,flush=True);e.close()

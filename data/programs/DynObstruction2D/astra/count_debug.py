from env_client import make_env
from approach import GeneratedApproach
import sys,time
seed=int(sys.argv[1]);count=int(sys.argv[2]);e=make_env();s,info=e.reset(seed=seed,options={'object_count':count});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);start=time.time();print('reset',info,flush=True)
for k in range(400):
 s,r,t,tr,i=e.step(p.get_action(s))
 if k%20==0 or t:
  o=s.get_object_from_name('target_block');ro=s.get_object_from_name('robot');sf=s.get_object_from_name('target_surface')
  print(k,p.phase,round(time.time()-start,1),'target',[round(s.get(o,f),3) for f in ['x','y','theta','width','height','held']],'robot',[round(s.get(ro,f),3) for f in ['x','y','theta']],'goal',s.get(sf,'x'),flush=True)
 if t or tr:break
e.close()

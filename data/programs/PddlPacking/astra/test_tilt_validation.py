from env_client import make_env
from candidate_tilt_validation import GeneratedApproach
import time,sys
startseed=int(sys.argv[1]) if len(sys.argv)>1 else 0
endseed=int(sys.argv[2]) if len(sys.argv)>2 else startseed+20
count=int(sys.argv[3]) if len(sys.argv)>3 else 0
for seed in range(startseed,endseed):
 e=make_env();s,info=e.reset(seed=seed,options={'object_count':count} if count else None);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);start=time.time();t=False
 for i in range(e.max_steps):
  s,_,t,tr,_=e.step(p.get_action(s))
  if t or tr:break
 print('RESULT',seed,info['object_count'],i+1,t,round(time.time()-start,2),'stage',p.stage,'stuck',p.stuck,flush=True);e.close()

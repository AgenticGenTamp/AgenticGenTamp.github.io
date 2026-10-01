from env_client import make_env
from approach import GeneratedApproach
import sys,time,json
for seed in [int(x) for x in sys.argv[1:]] or [0]:
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);start=time.time();last='';reward=0
 for k in range(e.max_steps):
  a=p.get_action(s);s,r,t,tr,info=e.step(a);reward+=r
  if p.phase!=last or t:
   print(seed,k,p.phase,'cube',p.cube.name,'xyz',p.xyz(s,p.cube).round(3),'bin',p.xyz(s,p.bin).round(3),'r',r,'term',t,flush=True);last=p.phase
  if t or tr:break
 print('RESULT',seed,k+1,t,tr,reward,round(time.time()-start,2),flush=True);e.close()

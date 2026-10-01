from env_client import make_env
from grasp import HookGrasp
for seed in [3,4]:
 e=make_env();s,i=e.reset(seed=seed);p=HookGrasp(e.action_space,e.observation_space);p.reset(s);h=s.get_object_from_name('hook');r=s.get_object_from_name('robot')
 phase=-1
 for k in range(220):
  if p.phase!=phase or k%20==0:
   print(seed,k,p.phase,'r',*[round(s.get(r,f),3) for f in ['x','y','theta','finger_gap']],'h',*[round(s.get(h,f),3) for f in ['x','y','theta','held']],flush=True);phase=p.phase
  s,rew,d,tr,i=e.step(p.get_action(s))
  if s.get(h,'held'):break
 e.close()

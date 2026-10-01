from env_client import make_env
from policy_variants import GeneratedApproach
for angle in (0.,-.16):
 e=make_env();s,i=e.reset(seed=101,options={'object_count':1});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);p.scoop_theta=angle
 for step in range(e.max_steps):
  phase=p.phase;a=p.get_action(s);s,_,term,trunc,_=e.step(a)
  if phase!=p.phase or term:
   print('angle',angle,'step',step+1,'phase',p.phase,'robot',*[round(s.get(p.robot,f),4) for f in ('x','y','theta')],'small',[(round(s.get(o,'x'),4),round(s.get(o,'y'),4)) for o in p.smalls],flush=True)
  if term or trunc:break
 print('RESULT',angle,term,step+1,flush=True);e.close()

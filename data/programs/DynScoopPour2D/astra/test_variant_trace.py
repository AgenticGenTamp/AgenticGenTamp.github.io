from env_client import make_env
from policy_variants import GeneratedApproach
import sys,time,json
seed=int(sys.argv[1]) if len(sys.argv)>1 else 42
limit=int(sys.argv[2]) if len(sys.argv)>2 else 250
E=make_env(); s,info=E.reset(seed=seed, options={"object_count":int(sys.argv[3])} if len(sys.argv)>3 else None)
p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,info)
start=time.monotonic()
for t in range(limit):
 a=p.get_action(s);s,r,term,trunc,i=E.step(a)
 if t%20==0 or term:
  objects=[s.get_object_from_name(n) for n in s.get_object_names()]
  def f(o):return [round(s.get(o,k),3) for k in ('x','y','theta')]
  print(t,'phase',p.phase,'robot',f(p.robot),'hook',f(p.hook),'held',s.get(p.hook,'held'),'smalls',[(o.name,f(o)[:2]) for o in objects if o.type.name.startswith('small')],flush=True)
 if term or trunc:break
print('result',seed,t+1,term,trunc,round(time.monotonic()-start,2),flush=True)
E.close()

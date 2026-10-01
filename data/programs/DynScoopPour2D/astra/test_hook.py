from env_client import make_env
from hook_policy import HookPolicy
import sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 101
count=int(sys.argv[2]) if len(sys.argv)>2 else 1
E=make_env();s,i=E.reset(seed=seed,options={'object_count':count});p=HookPolicy(E.observation_space);p.reset(s)
for t in range(1000):
 old=p.phase;s,re,term,trunc,i=E.step(p.act(s))
 if t%20==0 or p.phase!=old or term:
  print(t,p.phase,'r',[round(s.get(p.r,f),3) for f in ('x','y','theta','finger_gap')],'h',[round(s.get(p.h,f),3) for f in ('x','y','theta','held')],'small',[(round(s.get(o,'x'),3),round(s.get(o,'y'),3)) for typ in ('small_square','small_circle') for o in s.get_objects(E.observation_space.get_type(typ))],flush=True)
 if term or trunc:break
print('RESULT',term,t+1,flush=True);E.close()

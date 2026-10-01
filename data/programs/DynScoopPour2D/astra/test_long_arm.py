from env_client import make_env
from long_arm import GeneratedApproach
import sys,json,time
for n in map(int,sys.argv[1:] or [1,2,3,5,10,20,30,50,80]):
 E=make_env();s,info=E.reset(seed=100+n,options={'object_count':n});p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,info);start=time.monotonic()
 for t in range(1000):
  s,r,term,trunc,i=E.step(p.get_action(s))
  if term or trunc:break
 pts=[(s.get(o,'x'),s.get(o,'y')) for typ in ('small_circle','small_square') for o in s.get_objects(E.observation_space.get_type(typ))]
 print(json.dumps({'requested':n,'actual':len(pts),'ok':term,'steps':t+1,'right':sum(x>1.8 for x,y in pts),'time':round(time.monotonic()-start,2),'xy':[(round(x,3),round(y,3)) for x,y in pts] if not term else []}),flush=True)
 E.close()

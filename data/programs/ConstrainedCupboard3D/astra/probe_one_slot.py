from env_client import make_env
from approach import GeneratedApproach
import sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
slot=int(sys.argv[2]) if len(sys.argv)>2 else 3
count=int(sys.argv[3]) if len(sys.argv)>3 else 3
e=make_env();s,i=e.reset(seed=seed,options={'object_count':count});a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,i)
a.fixtures=[a.fixtures[slot]];last=None
for t in range(1000):
 s,r,te,tr,i=e.step(a.get_action(s));key=(a.index,a.stage,len(a.queue))
 if key!=last or te:
  print(t,key,te,[(o.name,a.xyz(s,o).round(3).tolist()) for o in a.objects],flush=True);last=key
 if te or tr:break
e.close()

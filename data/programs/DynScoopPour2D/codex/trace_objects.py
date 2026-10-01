import sys
from env_client import make_env
from approach import GeneratedApproach
e=make_env();s,info=e.reset(seed=int(sys.argv[1]) if len(sys.argv)>1 else 0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
names=[n for n in s.get_object_names() if n.startswith('small_')]
g=lambda n,f:float(s.get(s.get_object_from_name(n),f))
old=-1
for t in range(700):
 s,r,d,tr,i=e.step(p.get_action(s))
 if p.stage!=old and p.stage in (6,7,8,9,11,12,13,14):
  vals=sorted((g(n,'x'),g(n,'y')) for n in names)
  print(t,p.stage,'right',sum(x>2 for x,y in vals),'coords',[(round(x,2),round(y,2)) for x,y in vals],flush=True)
 old=p.stage
e.close()

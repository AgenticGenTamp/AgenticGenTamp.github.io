from env_client import make_env
from approach import GeneratedApproach
import sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 26
count=int(sys.argv[2]) if len(sys.argv)>2 else 10
e=make_env();s,i=e.reset(seed=seed,options={'object_count':count});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
for o in s.get_objects(p.ct):print(o.name,[round(s.get(o,'pose_'+f),5) for f in 'xyz'],flush=True)
for k in range(80):
 a=p.get_action(s);s,r,t,tr,i=e.step(a)
 if k<20 or k%10==0: print(k,'a',a.round(3).tolist(),'robot',[round(s.get(p.r,f),4) for f in p.fields],'active',s.get(p.r,'grasp_active'),'cube',[round(s.get(p.c,'pose_'+f),4) for f in 'xyz'],flush=True)
 if t or tr:print('DONE',k+1,t,tr);break
e.close()

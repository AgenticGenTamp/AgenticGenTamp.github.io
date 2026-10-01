from env_client import make_env
from approach import GeneratedApproach
import sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0;count=int(sys.argv[2]) if len(sys.argv)>2 else 3
e=make_env();s,i=e.reset(seed=seed,options={'object_count':count});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
for k in range(160):
 s,r,t,x,i=e.step(p.get_action(s))
 if k%10==0 or t:
  ro=s.get_object_from_name('robot'); print(k,p.phase,p.target,'base',round(s.get(ro,'pos_base_x'),3),round(s.get(ro,'pos_base_y'),3),'j2',round(s.get(ro,'joint_2'),3),'hold',s.get(ro,'grasp_active'),flush=True)
 if t:break
e.close()

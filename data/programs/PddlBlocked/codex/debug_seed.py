from env_client import make_env
from approach import GeneratedApproach
import sys
e=make_env();s,i=e.reset(seed=int(sys.argv[1]));p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);last=-1
for k in range(200):
 s,r,t,tr,i=e.step(p.get_action(s))
 if p.stage!=last or k%20==0:
  last=p.stage;print(k,'stage',last,'base',p.robot(s),'q2',p.g(s,'robot','joint_2'),'held',p.g(s,'robot','grasp_active'),'green',[p.g(s,'green0','pose_'+x) for x in 'xyz'],'route',p.route[:1])
 if t:print('WIN',k);break
e.close()

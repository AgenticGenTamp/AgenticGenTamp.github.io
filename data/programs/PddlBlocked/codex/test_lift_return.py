from env_client import make_env
from approach import GeneratedApproach
for seed in [5,15,23,27,37]:
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
 while p.stage<5:s,*_=e.step(p.get_action(s))
 t=p.target(p.green)
 for k in range(10):s,*_=e.step(p.motion(s,t,1,lift=True))
 print(seed,p.robot(s),'target',t)
 for k in range(3):s,*_=e.step(p.motion(s,t,1))
 a=p.motion(s,t,-1);s,*_=e.step(a);print('held',p.g(s,'robot','grasp_active'),'q2',p.g(s,'robot','joint_2'))
 e.close()

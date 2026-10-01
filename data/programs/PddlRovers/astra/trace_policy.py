import sys
from env_client import make_env
from approach import GeneratedApproach
E=make_env();s,info=E.reset(seed=int(sys.argv[1]));a=GeneratedApproach(E.action_space,E.observation_space,{});a.reset(s,info)
print('obs',a.obs)
for t in range(100):
 act=a.get_action(s);s,_,term,_,_=E.step(act)
 print(t,[a.xy(s,r).round(3).tolist() for r in a.rovers],act.round(4).tolist(),a.goal)
 if term:break
E.close()

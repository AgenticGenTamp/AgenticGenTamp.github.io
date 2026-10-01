from env_client import make_env
from approach import GeneratedApproach
import sys
for z in map(int,sys.argv[1:]):
 e=make_env();s,i=e.reset(seed=z);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);print(z,p.high_down,p.use_elbow,p.dynamic_tool);e.close()

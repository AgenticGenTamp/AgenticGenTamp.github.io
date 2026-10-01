from env_client import make_env
from prototype_agent import GeneratedApproach
import numpy as np, sys
for seed in map(int,sys.argv[1:]):
 e=make_env();s,_=e.reset(seed=seed);u=np.array([np.cos(s[11]),np.sin(s[11])]);n=np.array([u[1],-u[0]]);tip=s[9:11]+s[19]*n;g=s[29:31]-s[20:22];g/=np.linalg.norm(g)
 p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,{})
 point=s[9:11] if p.use_elbow else tip; pred=p.grasp_base+s[20:22]-.1*g-point
 print(seed,'r',s[:2],'gb',p.grasp_base,'pred',pred,'h',s[9:11],'th',s[11],'tip',tip,'mov',s[20:22],'tgt',s[29:31],'g',g,'dot',n@g,'dy elbow/tip behind',(s[20:22]-.1*g-s[9:11])[1],(s[20:22]-.1*g-tip)[1]);e.close()

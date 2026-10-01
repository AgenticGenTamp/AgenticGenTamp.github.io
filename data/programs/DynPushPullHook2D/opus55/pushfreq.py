import os
from env_client import make_env
from approach import GeneratedApproach
e=make_env(); c=0; L=[]
for s in range(200):
    o,i=e.reset(seed=s)
    ap=GeneratedApproach(e.action_space,e.observation_space,{}); ap.reset(o,i); ap.get_action(o)
    if ap.phase=='push': c+=1; L.append(s)
print(c, L)

import json
import numpy as np
from env_client import make_env

e=make_env()
poses={'shoulder_down': {1:0.6}, 'shoulder_up':{1:-1.0},'elbow_open':{3:-1.6},'elbow_closed':{3:-3.1},'wrist_down':{5:-1.5}}
for name,changes in poses.items():
 o,_=e.reset(seed=0);target=o[96:103].copy()
 for j,v in changes.items():target[j]=v
 for k in range(35):
  a=np.zeros(11,np.float32);a[3:10]=np.clip((target-o[96:103])*2,-.1,.1);a[10]=1
  o,*_=e.step(a)
 with open('pose_'+name+'.json','w') as f:json.dump(o.tolist(),f)
 print(name, np.round(o[96:103],3),flush=True)
e.close()

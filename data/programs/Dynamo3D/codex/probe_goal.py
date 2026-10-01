import sys
import numpy as np
from env_client import make_env

seed=int(sys.argv[1]); targets=[tuple(map(float,x.split(','))) for x in sys.argv[2:]]
env=make_env(); s,info=env.reset(seed=seed); rob=s.get_object_from_name('robot')
print('start',info,flush=True)
k=0
for tx,ty in targets:
  for _ in range(120):
    x=float(s.get(rob,'pos_base_x')); y=float(s.get(rob,'pos_base_y'))
    d=np.array([tx-x,ty-y]);
    if np.linalg.norm(d)<.03: break
    a=np.zeros(11,np.float32); a[:2]=np.clip(d*.15,-.1,.1)
    s,r,t,tr,inf=env.step(a); k+=1
    if k%10==0 or t or tr: print(k,r,t,tr,round(float(s.get(rob,'pos_base_x')),3),round(float(s.get(rob,'pos_base_y')),3),flush=True)
    if t or tr: env.close(); raise SystemExit
env.close(); print('end')

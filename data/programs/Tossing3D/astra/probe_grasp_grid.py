import numpy as np
from env_client import make_env
from approach import GeneratedApproach
from concurrent.futures import ThreadPoolExecutor

def trial(mount,xoff):
 e=make_env()
 try:
  s,info=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{})
  p.mount=mount;p.reset(s,info);c=p.cube;r=p.robot
  initial=np.array([s.get(c,f) for f in ['x','y','z']]);maxz=0
  for t in range(140):
   a=p.get_action(s)
   if p.t<55:a[0]=np.clip(s.get(c,'x')-.55-xoff-s.get(r,'pos_base_x'),-.1,.1)
   s,re,done,tr,info=e.step(a)
   if t>110:maxz=max(maxz,s.get(c,'z'))
  final=np.array([s.get(c,f) for f in ['x','y','z']])
  return {'mount':mount,'xoff':xoff,'maxz':round(maxz,4),'final':final.round(4).tolist(),'delta':(final-initial).round(4).tolist()}
 finally:e.close()
with ThreadPoolExecutor(max_workers=4) as pool:
 for out in pool.map(lambda args:trial(*args),[(m,x) for m in [.35,.45,.55,.65] for x in [0,.05,.1,-.05]]):print(out,flush=True)

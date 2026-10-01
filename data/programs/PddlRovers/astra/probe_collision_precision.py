from env_client import make_env
import numpy as np

e=make_env();s,_=e.reset(seed=0);r=s.get_object_from_name('rover0')
def move(x,y):
 global s
 for _ in range(40):
  old=np.array([s.get(r,'x'),s.get(r,'y')]); d=np.clip(np.array([x,y])-old,-.2,.2);a=np.zeros(8);a[:2]=d;s,*_=e.step(a)
  new=np.array([s.get(r,'x'),s.get(r,'y')])
  if np.linalg.norm(new-old)<1e-6 or np.linalg.norm(new-[x,y])<1e-6:break
 print('target',x,y,'actual',new,flush=True)
move(1,-.1987)
for x in [1,.95,.94,.93,.92,.91,.9,.89,.88,.87,.86,.85,.84]:move(x,-.1987)
move(1,-.1987);move(1,-2.25)
for y in [-2.26,-2.27,-2.28]:move(1,y)
move(.4,-2.25)
for x in [.26,.25,.249,.24,.23,.225,.224,.22]:move(x,-2.25)
e.close()

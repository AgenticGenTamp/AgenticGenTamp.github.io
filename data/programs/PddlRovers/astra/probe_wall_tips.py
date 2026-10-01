from env_client import make_env
import numpy as np
for y in [-2.27,-2.26,-2.2,-2.1,-2.0,1.5]:
 e=make_env();s,_=e.reset(seed=0);r=s.get_object_from_name('rover0')
 def move(x,y):
  global s
  for _ in range(40):
   old=np.array([s.get(r,'x'),s.get(r,'y')]);d=np.clip(np.array([x,y])-old,-.2,.2);a=np.zeros(8);a[:2]=d;s,*_=e.step(a);new=np.array([s.get(r,'x'),s.get(r,'y')])
   if np.linalg.norm(new-old)<1e-6 or np.linalg.norm(new-[x,y])<1e-6:break
  return new
 move(1,y);move(.24,y)
 print(y,'start',[s.get(r,'x'),s.get(r,'y')], 'end',move(-.4,y),flush=True)
 e.close()

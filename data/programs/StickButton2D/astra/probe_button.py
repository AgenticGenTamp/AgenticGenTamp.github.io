from env_client import make_env
import numpy as np, math
E=make_env();s,_=E.reset(seed=0);r=s.get_object_from_name('robot');b=s.get_object_from_name('button2')
def drive(x,y,t,j):
 global s
 for k in range(100):
  old=np.array([s.get(r,f) for f in ('x','y','theta','arm_joint')])
  d=np.array([x,y,t,j])-old;d[2]=(d[2]+math.pi)%(2*math.pi)-math.pi
  a=np.r_[np.clip(d,[-.05,-.05,-.19634954,-.1],[.05,.05,.19634954,.1]),0].astype(np.float32)
  s,rw,te,tr,_=E.step(a)
  new=np.array([s.get(r,f) for f in ('x','y','theta','arm_joint')])
  if np.max(abs(new-old))<1e-6:break
 print('pose',new,'color',[s.get(b,f) for f in ('color_r','color_g','color_b')], 'reward',rw)
x,y=s.get(b,'x'),s.get(b,'y')
for off in [.3,.25,.2,.15,.1]:
 drive(x,y+off,-math.pi/2,.2)
E.close()

from env_client import make_env
import numpy as np
E=make_env();s,_=E.reset(seed=1);r=s.get_object_from_name('robot');b=s.get_object_from_name('target_block')
def vals(o):return [round(s.get(o,k),5) for k in ['x','y','theta']]
def go(x,y,vac=0):
 global s
 for _ in range(60):
  dx=x-s.get(r,'x');dy=y-s.get(r,'y')
  if max(abs(dx),abs(dy))<1e-5:break
  s,*_=E.step(np.array([np.clip(dx,-.05,.05),np.clip(dy,-.05,.05),0,0,vac]))
x=s.get(b,'x')+s.get(b,'width')/2
go(x,.7)
print('START',vals(r),vals(b))
for i in range(55):
 old=s
 s,rew,te,tr,info=E.step([0,-.005,0,0,1])
 print(i,vals(r),vals(b),s.get(r,'vacuum'),info)
 if abs(s.get(b,'y')-.1)>.001:break
for i in range(10):
 s,*_=E.step([0,.02,0,0,1]);print('UP',i,vals(r),vals(b))
E.close()

from env_client import make_env
import numpy as np

def vals(s,e):
 for typ in e.observation_space.types:
  obs=s.get_objects(typ)
  if obs:
   for o in obs:
    print(str(o), {f:round(s.get(o,f),4) for f in e.observation_space.type_features[typ]})

def rv(s,e):
 r=next(iter(s.get_objects(e.observation_space.get_type('crv_robot'))))
 return {f:round(s.get(r,f),4) for f in ['x','y','theta','arm_joint','arm_length']}

e=make_env();s,i=e.reset(seed=0);print('initial');vals(s,e);print('info',i,'max',e.max_steps)
for label,a,n in [('extend',[0,0,0,.1,0],12),('retract',[0,0,0,-.1,0],14),('right',[.05,0,0,0,0],30),('up',[0,.05,0,0,0],30),('left',[-.05,0,0,0,0],30),('down',[0,-.05,0,0,0],30),('turn',[0,0,.19634954,0,0],36)]:
 for k in range(n):
  s,re,t,tr,inf=e.step(np.array(a,dtype=np.float32))
  if k in [0,1,n-1]: print(label,k,rv(s,e),'r',re,'done',t,tr)
e.close()

e=make_env();s,_=e.reset(seed=0)
for axis,target in [(0,3.4),(1,1.15),(0,.1),(1,.1)]:
 for _ in range(100):
  r=next(iter(s.get_objects(e.observation_space.get_type('crv_robot'))))
  cur=s.get(r,'x' if axis==0 else 'y');a=np.zeros(5,dtype=np.float32);a[axis]=np.clip(target-cur,-.05,.05)
  s,*_=e.step(a)
  if abs(cur-target)<1e-6:break
 print('target',axis,target,'actual',rv(s,e))
 for delta in [.001,-.001]:
  a=np.zeros(5,dtype=np.float32);a[axis]=delta;s,*_=e.step(a);print('delta',delta,rv(s,e))
e.close()

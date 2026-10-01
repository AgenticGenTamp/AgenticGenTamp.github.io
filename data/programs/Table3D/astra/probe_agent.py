from env_client import make_env
import numpy as np

def dump(e,s):
 for ty in e.observation_space.types:
  for ob in s.get_objects(ty):
   print(str(ob), {f:round(float(s.get(ob,f)),5) for f in e.observation_space.type_features[ty]})

e=make_env();s,_=e.reset(seed=0)
print('INITIAL');dump(e,s)
r=next(iter(s.get_objects(e.observation_space.get_type('Kinematic3DRobot'))))
fs=e.observation_space.type_features[e.observation_space.get_type('Kinematic3DRobot')]
for i in range(11):
 s,_=e.reset(seed=0)
 a=np.zeros(11);a[i]=.2 if i<10 else -1
 before=np.array([s.get(r,f) for f in fs])
 s,rw,te,tr,info=e.step(a)
 after=np.array([s.get(r,f) for f in fs])
 print('ACTION',i,'CHANGES',{f:round(float(y-x),6) for f,x,y in zip(fs,before,after) if abs(y-x)>1e-8},'info',info)
e.close()
e=make_env();s,_=e.reset(seed=1);r=next(iter(s.get_objects(e.observation_space.get_type('Kinematic3DRobot'))))
for val in [1,0,-1,1,-1]:
 a=np.zeros(11);a[-1]=val;s,*_=e.step(a)
 print('FINGER',val,s.get(r,'finger_state'))
for i in range(10):
 a=np.zeros(11);a[0]=.4;s,*_=e.step(a)
 print('BASESTEP',i,s.get(r,'pos_base_x'))
e.close()

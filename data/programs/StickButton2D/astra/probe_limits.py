from env_client import make_env
import numpy as np, math

def move(e,s,goal):
 r=next(iter(s.get_objects(e.observation_space.get_type('crv_robot'))))
 for _ in range(80):
  c=np.array([s.get(r,f) for f in ['x','y','theta','arm_joint']]);d=np.array(goal)-c;d[2]=(d[2]+math.pi)%(2*math.pi)-math.pi
  if np.max(abs(d))<1e-5:break
  a=np.zeros(5,dtype=np.float32);a[:4]=np.clip(d,[-.05,-.05,-.19634954,-.1],[.05,.05,.19634954,.1]);s,*_=e.step(a)
 return s
for theta in [0,math.pi/2,math.pi,-math.pi/2]:
 e=make_env();s,_=e.reset(seed=4);s=move(e,s,[1,.5,theta,.2]);r=next(iter(s.get_objects(e.observation_space.get_type('crv_robot'))))
 out=[]
 for axis,sgn in [(0,-1),(0,1),(1,-1),(1,1)]:
  s=move(e,s,[1,.5,theta,.2]);a=np.zeros(5,dtype=np.float32);a[axis]=sgn*.05
  for _ in range(80):s,*_=e.step(a)
  a[axis]=sgn*.001
  for _ in range(50):s,*_=e.step(a)
  out.append(round(s.get(r,'x' if axis==0 else 'y'),4))
 print('angle',theta,'bounds',out)
 e.close()

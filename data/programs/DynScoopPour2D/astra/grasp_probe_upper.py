from env_client import make_env
import numpy as np,math

def trial(seed,xoff,yoff,theta):
 e=make_env();s,_=e.reset(seed=seed);r=s.get_objects(e.observation_space.get_type('kin_robot'))[0];h=s.get_objects(e.observation_space.get_type('hook'))[0];hx,hy=s.get(h,'x'),s.get(h,'y');steps=0
 def snap(tag):print(seed,xoff,yoff,theta,tag,steps,'robot',[round(s.get(r,f),3) for f in ('x','y','theta','arm_joint','finger_gap')],'hook',[round(s.get(h,f),3) for f in ('x','y','theta','held')],flush=True)
 def move(axis,val,n=150):
  nonlocal s,steps
  f=('x','y','theta','arm_joint','finger_gap')[axis]
  for _ in range(n):
   d=val-s.get(r,f)
   if axis==2:d=(d+math.pi)%(2*math.pi)-math.pi
   if abs(d)<.0005:break
   a=np.zeros(5);a[axis]=d
   s,*_=e.step(np.clip(a,[-.0299,-.0299,-.0979,-.0799,-.0149],[.0299,.0299,.0979,.0799,.0149]));steps+=1
   if s.get(h,'held'):break
 move(1,2.4);move(0,2.8);move(2,theta);move(0,hx+xoff);move(1,hy+yoff);snap('approach');move(4,.08,20);snap('closed')
 if s.get(h,'held'):
  for k in range(30):s,*_=e.step(np.array([0,.0299,0,0,0]));steps+=1
  snap('lifted')
 e.close()
if __name__=='__main__':
 trial(0,-.15,.65,-math.pi/3)
 trial(0,-.22,.65,-math.pi/4)
 trial(0,0,.7,-math.pi/2)

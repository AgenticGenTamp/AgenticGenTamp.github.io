from env_client import make_env
import numpy as np,math

def trial(speed,y):
 e=make_env();s,info=e.reset(seed=101,options={'object_count':1});r=s.get_objects(e.observation_space.get_type('kin_robot'))[0];small=[o for typ in ('small_circle','small_square') for o in s.get_objects(e.observation_space.get_type(typ))];steps=0;done=False
 def step(a):
  nonlocal s,steps,done
  s,_,done,_,_=e.step(np.clip(a,[-.0299,-.0299,-.0979,-.0799,-.0149],[.0299,.0299,.0979,.0799,.0149]));steps+=1
 def snap(tag):print(speed,y,tag,steps,'done',done,'robot',[round(s.get(r,f),3) for f in ('x','y','theta','arm_joint','finger_gap')],'small',[[round(s.get(o,f),3) for f in ('x','y','vx','vy')]for o in small],flush=True)
 def move(axis,value,n=200,limit=.0299):
  f=('x','y','theta','arm_joint','finger_gap')[axis]
  for _ in range(n):
   d=value-s.get(r,f)
   if axis==2:d=(d+math.pi)%(2*math.pi)-math.pi
   if abs(d)<.001 or done:break
   a=np.zeros(5);a[axis]=np.clip(d,-limit,limit) if axis in (0,1) else d;step(a)
 move(1,2.5);move(2,-math.pi/2);move(0,s.get(small[0],'x'));move(1,y);snap('positioned')
 move(4,.08,25);snap('pinched')
 move(1,.65,90,limit=speed);snap('partiallift')
 move(1,2.,160,limit=speed);snap('lifted');move(0,2.5);move(4,.25,25);snap('released')
 e.close()
if __name__=='__main__':
 trial(.01,.34)
 trial(.0299,.36)

from env_client import make_env
import numpy as np,math

def trial(mode):
 e=make_env();s,info=e.reset(seed=101,options={'object_count':1});r=s.get_objects(e.observation_space.get_type('kin_robot'))[0];small=[o for typ in ('small_circle','small_square') for o in s.get_objects(e.observation_space.get_type(typ))];steps=0;done=False
 def step(a):
  nonlocal s,steps,done
  s,_,done,_,_=e.step(np.clip(a,[-.0299,-.0299,-.0979,-.0799,-.0149],[.0299,.0299,.0979,.0799,.0149]));steps+=1
 def snap(tag):print(mode,tag,steps,'done',done,'robot',[round(s.get(r,f),3) for f in ('x','y','theta','arm_joint','finger_gap')],'small',[[round(s.get(o,f),3) for f in ('x','y','vx','vy')]for o in small],flush=True)
 def move(axis,value,n=120):
  f=('x','y','theta','arm_joint','finger_gap')[axis]
  for _ in range(n):
   d=value-s.get(r,f)
   if axis==2:d=(d+math.pi)%(2*math.pi)-math.pi
   if abs(d)<.001 or done:break
   a=np.zeros(5);a[axis]=d;step(a)
 move(1,2.6);move(2,math.pi/2);move(0,.23);move(1,.2);snap('start')
 move(0,1.51,65);snap('compressed')
 if mode==0:
  move(1,1.85);snap('lift');move(0,2.5);snap('carry')
 elif mode==1:
  for j in range(10):
   move(0,1.42,10);move(0,1.52,10)
   if done:break
  snap('pulses');move(1,1.85);move(0,2.5);snap('liftcarry')
 else:
  for j in range(15):
   move(2,math.pi/2-.25,5);move(2,math.pi/2+.25,10);move(0,1.52,3)
   if done:break
  snap('rock');move(1,1.85);move(0,2.5);snap('liftcarry')
 e.close()
if __name__=='__main__':
 for mode in range(3):trial(mode)

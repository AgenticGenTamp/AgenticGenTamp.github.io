from probe_tilt import pose
from kinematics import HOME
from env_client import make_env
import numpy as np

e=make_env();s,inf=e.reset(seed=0,options={'object_count':2});r=s.get_object_from_name('robot');cup=s.get_object_from_name('cupboard_1');z=float(s.get(cup,'z'));steps=0;done=False
objects=sorted(n for n in s.get_object_names() if n.startswith('cube'))
def positions():return {n:np.round([s.get(s.get_object_from_name(n),f) for f in ['x','y','z']],4).tolist() for n in objects}
def move(q,b,g,n):
 global s,steps,done
 for i in range(n):
  cur=np.array([s.get(r,'pos_arm_joint'+str(j)) for j in range(1,8)]);base=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
  a=np.r_[np.clip(b-base,-.1,.1),np.clip(2*(q-cur),-.1,.1),g].astype(np.float32)
  s,rew,t,tr,inf=e.step(a);steps+=1
  if t or rew!=-1: print('FOUND',steps,'reward',rew,'term',t,'cubes',positions(),flush=True);done|=t
  if done:break
 return positions()
up=pose(.6,z+.64,-np.pi/2);down=pose(.5,-.04)
for idx,name in enumerate(objects):
 o=s.get_object_from_name(name);b0=np.array([s.get(o,'x')-.625,s.get(o,'y')-.001,0.])
 print(name,'travel',move(HOME,b0,0,65),flush=True)
 move(down,b0,0,95);print(name,'grasp',move(down,b0,1,15),flush=True)
 print(name,'lift',move(up,b0,1,115),flush=True)
 dest=np.array([1.5-.6-.125,(-.15 if idx==0 else .15)-.001,0.]);print(name,'insert',move(up,dest,1,90),flush=True)
 print(name,'drop',move(up,dest,0,35),flush=True)
 dest[0]-=.4;move(up,dest,0,25)
 if done:break
e.close()

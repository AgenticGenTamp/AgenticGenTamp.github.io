from env_client import make_env
import numpy as np

def dump(s):
 for name in sorted(s.get_object_names()):
  o=s.get_object_from_name(name)
  fs=['x','y','theta','base_radius','arm_joint','arm_length','vacuum','gripper_height','gripper_width'] if o.type.name=='crv_robot' else ['x','y','theta','width','height','static']
  print(name, {f:round(s.get(o,f),4) for f in fs})

def main():
 e=make_env()
 for seed in [0,1,42]:
  s,_=e.reset(seed=seed);print('SEED',seed);dump(s)
 s,_=e.reset(seed=0);r=s.get_object_from_name('robot')
 for a,n in [([0,0,0,-.1,0],10),([-.05,0,0,0,0],15),([0,-.05,0,0,0],15),([0,0,0,.1,0],15),([0,0,.19634954,0,0],32)]:
  for i in range(n):
   old=[s.get(r,f) for f in ['x','y','theta','arm_length']]
   s,*_=e.step(np.array(a,np.float32))
   new=[s.get(r,f) for f in ['x','y','theta','arm_length']]
   if i==0 or i==n-1 or np.linalg.norm(np.array(new)-old)<1e-6:print('MOVE',a,i,np.round(old,4),np.round(new,4))
 e.close()
if __name__=='__main__': main()

def contact_probe(joint=.1,vac=0):
 e=make_env();s,_=e.reset(seed=42);r=s.get_object_from_name('robot');t=s.get_object_from_name('target_block')
 def step(a):
  nonlocal s
  s,*_=e.step(np.array(a,np.float32))
 def goto(x,y,th):
  for _ in range(100):
   d=np.array([x-s.get(r,'x'),y-s.get(r,'y'),(th-s.get(r,'theta')+np.pi)%(2*np.pi)-np.pi])
   if max(abs(d))<1e-5: break
   step([*np.clip(d,[-.05,-.05,-.196],[.05,.05,.196]),0,0])
 goto(1.15,.8,0);goto(1.15,s.get(t,'y'),0);step([0,0,0,joint-.1,0])
 print('contact start',joint,vac);dump(s)
 for i in range(55):
  step([.01,0,0,0,vac])
  if i%5==0:print(i,round(s.get(r,'x'),4),round(s.get(t,'x'),4),round(s.get(r,'arm_joint'),4),s.get(r,'vacuum'))
 e.close()

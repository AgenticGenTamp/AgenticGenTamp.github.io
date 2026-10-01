from env_client import make_env
import numpy as np

def dump(e,s):
 for t in ('crv_robot','shelf','target_block'):
  ty=e.observation_space.get_type(t)
  for o in s.get_objects(ty):
   fs=['x','y','theta'] + ({'crv_robot':['base_radius','arm_joint','arm_length','gripper_height','gripper_width'],'shelf':['width','height','x1','y1','theta1','width1','height1'],'target_block':['width','height']}[t])
   print(str(o), {f:round(float(s.get(o,f)),5) for f in fs})
if __name__=='__main__':
 for seed in range(5):
  e=make_env();s,i=e.reset(seed=seed);print('SEED',seed,'INFO',i);dump(e,s);e.close()
 e=make_env();s,i=e.reset(seed=1)
 r=next(iter(s.get_objects(e.observation_space.get_type('crv_robot'))))
 for j in range(50):
  s,rew,term,trunc,info=e.step(np.array([0,.05,0,-.1,0],dtype=np.float32))
  if j%5==0:print('UP',j,{f:float(s.get(r,f)) for f in ['x','y','theta','arm_length','arm_joint']},rew,term,trunc,info)
 e.close()
 e=make_env();s,i=e.reset(seed=1)
 r=next(iter(s.get_objects(e.observation_space.get_type('crv_robot'))))
 for j in range(8):
  s,rew,term,trunc,info=e.step(np.array([0,0,0,.1,0],dtype=np.float32))
  print('EXTEND',j,{f:float(s.get(r,f)) for f in ['arm_length','arm_joint']})
 e.close()
 e=make_env();s,i=e.reset(seed=0)
 r=next(iter(s.get_objects(e.observation_space.get_type('crv_robot'))))
 def go(x,y,t):
  global s
  for _ in range(150):
   xx,yy,tt=[float(s.get(r,f)) for f in ['x','y','theta']]
   dt=(t-tt+np.pi)%(2*np.pi)-np.pi
   if max(abs(x-xx),abs(y-yy),abs(dt))<.001:break
   s,*_=e.step(np.array([np.clip(x-xx,-.05,.05),np.clip(y-yy,-.05,.05),np.clip(dt,-.196,.196),-.1,0],dtype=np.float32))
  print('GO',x,y,t,'GOT',{f:round(float(s.get(r,f)),4) for f in ['x','y','theta','arm_joint']})
 go(2.96,.5,-np.pi/2);go(.54,.5,-np.pi/2);go(.54,2.9,-np.pi/2)
 dump(e,s);e.close()
 e=make_env();s,i=e.reset(seed=0);r=next(iter(s.get_objects(e.observation_space.get_type('crv_robot'))));b=s.get_object_from_name('block1')
 go(2.96,.5,np.pi/2);go(.54,.5,np.pi/2);go(.54,2.2,np.pi/2)
 for j in range(120):
  s,rew,term,trunc,info=e.step(np.array([0,.005,0,0,1],dtype=np.float32))
  if j%10==0:print('PUSHVAC',j,'robot',float(s.get(r,'y')),'block',float(s.get(b,'y')),'vac',float(s.get(r,'vacuum')))
 dump(e,s);e.close()
 e=make_env();s,i=e.reset(seed=0);r=next(iter(s.get_objects(e.observation_space.get_type('crv_robot'))));b=s.get_object_from_name('block1')
 go(2.96,.5,np.pi/2);go(.54,.5,np.pi/2);go(.54,2.2,np.pi/2)
 for j in range(60):s,*_=e.step(np.array([0,.005,0,0,1],dtype=np.float32))
 for j in range(80):
  s,rew,term,trunc,info=e.step(np.array([0,0,0,.005,1],dtype=np.float32))
  if j%5==0:print('DEEP',j,'robot',float(s.get(r,'y')),'arm',float(s.get(r,'arm_joint')),'block',float(s.get(b,'y')))
 e.close()
 for seed in range(3):
  e=make_env();s,i=e.reset(seed=seed,options={'object_count':1});print('SINGLE',seed,i);dump(e,s)
  s,re,te,tr,inf=e.step(np.zeros(5,dtype=np.float32));print('NOOP',re,te,tr);e.close()
 e=make_env();s,i=e.reset(seed=0,options={'object_count':1});r=next(iter(s.get_objects(e.observation_space.get_type('crv_robot'))));b=s.get_object_from_name('block0')
 a=float(s.get(b,'theta'));n=np.array([-np.sin(a),np.cos(a)])
 def center():
  a=float(s.get(b,'theta'));return np.array([float(s.get(b,'x'))+.14*np.cos(a)-.02*np.sin(a),float(s.get(b,'y'))+.14*np.sin(a)+.02*np.cos(a)])
 p=center()-.54*n;go(*p,a+np.pi/2)
 for j in range(35):s,*_=e.step(np.array([0,0,0,.01,1],dtype=np.float32))
 print('SINGLE GRASP');dump(e,s)
 for j in range(15):
  dt=(np.pi/2-float(s.get(r,'theta'))+np.pi)%(2*np.pi)-np.pi
  s,*_=e.step(np.array([0,0,np.clip(dt,-.025,.025),0,1],dtype=np.float32))
 print('SINGLE ROTATED');dump(e,s)
 sh=next(iter(s.get_objects(e.observation_space.get_type('shelf'))));dest=np.array([float(s.get(sh,'x1'))+float(s.get(sh,'width1'))/2,2.68])
 for target in (np.array([dest[0],center()[1]]),dest):
  for j in range(150):
   delta=target-center()
   if max(abs(delta))<.001:break
   s,re,te,tr,inf=e.step(np.array([*np.clip(delta,-.02,.02),0,0,1],dtype=np.float32))
   if te:print('TERMINATED',center(),{f:float(s.get(b,f)) for f in ['x','y','theta']});break
  print('SINGLE CARRY target',target,'actual',center(),'term',te)
 e.close()

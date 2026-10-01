import numpy as np, math
from env_client import make_env

def grasp(seed=0,index='block4',dist=.8):
 env=make_env();s,_=env.reset(seed=seed)
 r=next(iter(s.get_objects(env.observation_space.get_type('crv_robot'))));b=s.get_object_from_name(index)
 def val(o,f):return float(s.get(o,f))
 bt=val(b,'theta'); bx=val(b,'x');by=val(b,'y');w=val(b,'width');h=val(b,'height')
 cx=bx+math.cos(bt)*w/2-math.sin(bt)*h/2;cy=by+math.sin(bt)*w/2+math.cos(bt)*h/2
 def move(x,y,t,a,v=0,n=100):
  nonlocal s
  stalled=0
  for i in range(n):
   delta=np.array([x-val(r,'x'),y-val(r,'y'),(t-val(r,'theta')+math.pi)%(2*math.pi)-math.pi,a-val(r,'arm_joint'),v])
   if np.max(np.abs(delta[:4]))<1e-5:break
   old=[val(r,f) for f in ('x','y','theta','arm_joint')]
   s,*_=env.step(np.clip(delta,env.action_space.low,env.action_space.high).astype(np.float32))
   new=[val(r,f) for f in ('x','y','theta','arm_joint')]
   stalled=stalled+1 if old==new else 0
   if stalled>=3:break
  return i
 move(val(r,'x'),val(r,'y'),0,.2)
 move(4.6,val(r,'y'),0,.2)
 move(4.6,cy,0,.2)
 move(4.6,cy,math.pi,.2)
 move(cx+dist,cy,math.pi,.2)
 for i in range(150):
  s,*_=env.step(np.array([0,0,0,.005,1],np.float32))
  if (val(b,'x'),val(b,'y'))!=(bx,by):print('firstchange',i,'arm',val(r,'arm_joint'),'block',val(b,'x'),val(b,'y'));break
 before=(val(b,'x'),val(b,'y'));rb=(val(r,'x'),val(r,'y'),val(r,'arm_joint'))
 for _ in range(5):s,*_=env.step(np.array([.02,0,0,0,1],np.float32))
 after=(val(b,'x'),val(b,'y'))
 print('dist',dist,'center',cx,cy,'robot',rb,'blockdelta',tuple(np.array(after)-before),'blockpre',before,flush=True)
 env.close()
if __name__=='__main__':
 for dist in [.6,.8,1.]:grasp(dist=dist)

def tunnel(delta,vac):
 env=make_env();s,_=env.reset(seed=0);r=next(iter(s.get_objects(env.observation_space.get_type('crv_robot'))));b=s.get_object_from_name('block4')
 def step(a):
  nonlocal s
  s,*_=env.step(np.array(a,np.float32))
 def val(o,f):return float(s.get(o,f))
 for _ in range(20):
  d=(0-val(r,'theta')+math.pi)%(2*math.pi)-math.pi
  step([0,0,np.clip(d,-.196,.196),0,0])
 for _ in range(10):step([.05,0,0,0,0])
 before=[val(r,f) for f in ('x','y','theta','arm_joint')]
 step([0,0,0,delta,vac]);after=[val(r,f) for f in ('x','y','theta','arm_joint')]
 step([.02,0,0,0,vac]);print('tunnel',delta,vac,'rb',before,'ra',after,'block',val(b,'x'),val(b,'y'),flush=True)
 env.close()
if __name__=='__main__':
 for delta,vac in [(.005,0),(.1,0),(.005,1),(.1,1)]:tunnel(delta,vac)

def normal_probe(tunnel=False):
 env=make_env();s,_=env.reset(seed=0);r=next(iter(s.get_objects(env.observation_space.get_type('crv_robot'))));b=s.get_object_from_name('block4')
 def val(o,f):return float(s.get(o,f))
 def step(a):
  nonlocal s
  s,*_=env.step(np.array(a,np.float32))
 t=val(b,'theta')-math.pi/2;u=np.array([math.cos(t),math.sin(t)]);v=np.array([math.cos(val(b,'theta')),math.sin(val(b,'theta'))]);p=np.array([val(b,'x'),val(b,'y')])+v*.14-u*.04;base=p-u*.6
 for _ in range(30):step([0,0,np.clip((t-val(r,'theta')+math.pi)%(2*math.pi)-math.pi,-.196,.196),0,0])
 for _ in range(30):step([np.clip(base[0]-val(r,'x'),-.05,.05),np.clip(base[1]-val(r,'y'),-.05,.05),0,0,0])
 old=(val(b,'x'),val(b,'y'));print('normal targetpoint',p,'base',base,'actual',[val(r,f) for f in ('x','y','theta')],flush=True)
 if tunnel:
  for _ in range(70):step([0,0,0,.005,0])
  print('before tunnel',val(r,'arm_joint'),flush=True)
  step([0,0,0,.1,0]);print('after tunnel',val(r,'arm_joint'),flush=True)
  env.close();return
 for i in range(100):
  step([0,0,0,.005,1]);new=(val(b,'x'),val(b,'y'))
  if old!=new:print('grasp arm',val(r,'arm_joint'),'moved',np.array(new)-old,'nominal contact=.6',flush=True);break
 env.close()
if __name__=='__main__':normal_probe()

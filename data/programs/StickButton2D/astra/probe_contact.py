from env_client import make_env
import numpy as np, math

env=make_env(); s,_=env.reset(seed=0)
r=s.get_object_from_name('robot'); st=s.get_object_from_name('stick')
def move(tx,ty,tt=math.pi/2,arm=.1,vac=0,n=100):
 global s
 for i in range(n):
  x,y,t,a=[s.get(r,k) for k in ['x','y','theta','arm_joint']]
  dt=(tt-t+math.pi)%(2*math.pi)-math.pi
  ac=[np.clip(tx-x,-.05,.05),np.clip(ty-y,-.05,.05),np.clip(dt,-.196,.196),np.clip(arm-a,-.1,.1),vac]
  s,rew,term,trunc,info=env.step(np.array(ac,dtype=np.float32))
  if sum(abs(z) for z in ac[:4])<1e-5:break
 print('robot',*[round(s.get(r,k),4) for k in ['x','y','theta','arm_joint','vacuum']], 'stick',*[round(s.get(st,k),4) for k in ['x','y','theta']], 'colors',[(n,s.get(s.get_object_from_name(n),'color_g')) for n in ['button0','button1','button2']])
sx,sy=s.get(st,'x'),s.get(st,'y')
move(.5,.7,tt=math.pi,arm=.2,vac=1)
move(.5,sy-.025,tt=math.pi,arm=.2,vac=1)
for x in np.arange(.4,.27,-.002):
 move(x,sy-.025,tt=math.pi,arm=.2,vac=1,n=1)
 if s.get(st,'x')!=sx: print('ATTACHED',x);break
move(.7,.8,tt=math.pi,arm=.2,vac=1)
move(2.1423,1.05,tt=math.pi,arm=.2,vac=1)
move(2.3566,1.05,tt=math.pi,arm=.2,vac=1)
env.close()

from env_client import make_env
import numpy as np, math
for tilt in [.15,.25,.35]:
 e=make_env();s,_=e.reset(seed=112);r=s.get_object_from_name('robot');o=s.get_object_from_name('target_block')
 def g(ob,f):return s.get(ob,f)
 def go(x,y,theta,arm=.48,gap=.32):
  global s
  for _ in range(80):
   a=[x-g(r,'x'),y-g(r,'y'),theta-g(r,'theta'),arm-g(r,'arm_joint'),gap-g(r,'finger_gap')]
   s,*_=e.step(np.clip(a,e.action_space.low*.999,e.action_space.high*.999))
   if max(abs(a[j]) for j in range(4))<.005:break
 go(g(r,'x'),1.3,g(r,'theta'),.24)
 go(1.1,1.3,math.pi+tilt)
 go(g(o,'x')+g(o,'width')/2+.53,g(o,'y')+.60*math.sin(tilt),math.pi+tilt)
 held=False
 for k in range(70):
  s,*_=e.step(np.array([-.005,0,0,0,-.001]))
  if g(o,'held'):held=True;break
  if k%20==0:print('progress',tilt,k,[round(g(o,f),3) for f in ['x','y','theta']],flush=True)
 print('RESULT',tilt,held,k,'obj',[round(g(o,f),3) for f in ['x','y','theta']],'robot',[round(g(r,f),3) for f in ['x','y','theta','finger_gap']],flush=True)
 e.close()

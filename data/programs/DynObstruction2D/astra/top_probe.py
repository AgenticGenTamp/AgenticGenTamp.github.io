from env_client import make_env
import numpy as np,math
for seed in range(4):
 for mode in [0,1,2,3]:
  e=make_env();s,_=e.reset(seed=seed);r=s.get_object_from_name('robot');o=s.get_object_from_name('target_block')
  def g(o,f):return s.get(o,f)
  def step(a):
   global s
   s,*_=e.step(np.clip(a,[-.049,-.049,-.19,-.099,-.019],[.049,.049,.19,.099,.019]))
  def go(x,y,theta):
   for k in range(100):
    d=[x-g(r,'x'),y-g(r,'y'),theta-g(r,'theta'),.24-g(r,'arm_joint'),.32-g(r,'finger_gap')]
    step(d)
    if max(abs(v) for v in d[:3])<.001:break
  go(g(r,'x'),1.65,g(r,'theta'));go(g(o,'x'),1.65,-math.pi/2)
  top=g(o,'y')+g(o,'height')/2
  go(g(o,'x'),top+.49,-math.pi/2)
  for k in range(80):
   # mode0 gap full negative tiny. mode1 close-fast; mode2 start narrower. mode3 alternate open/close
   step([np.clip(g(o,'x')-g(r,'x'),-.005,.005),-.005,0,0,(-.0001 if mode==0 else -.019) if mode<2 else (-.003 if mode==2 else (.019 if k%2 else -.019))])
   if g(o,'held') or g(o,'y')<0:break
  print(seed,mode,'step',k,'held',g(o,'held'),'obj',round(g(o,'x'),3),round(g(o,'y'),3),'base',round(g(r,'x'),3),round(g(r,'y'),3),'gap',round(g(r,'finger_gap'),3),flush=True)
  e.close()

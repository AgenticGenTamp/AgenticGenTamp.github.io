from env_client import make_env
import numpy as np
for g in [0,1]:
 e=make_env();s,i=e.reset(seed=0);robot=s.get_object_from_name('robot');cube=s.get_object_from_name('cube_0')
 for k in range(60):
  a=np.zeros(18);a[10]=g
  if k<10:a[0]=.04;a[1]=.025
  if k>=10 and k<35:a[4]=-.1;a[6]=.1
  s,r,t,tr,i=e.step(a)
  if k%10==9:print(g,k,'cube',[round(s.get(cube,f),3) for f in ['x','y','z']],'q',[round(s.get(robot,'pos_arm_joint'+str(j)),3) for j in range(1,8)])
 e.close()

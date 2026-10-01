from env_client import make_env
from approach import GeneratedApproach
import numpy as np
E=make_env();s,info=E.reset(seed=0);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,info);rb=p.robot;cube=s.get_object_from_name('cube1')
def move(base,z,grip,n):
 global s
 for _ in range(n):
  q=np.array([s.get(rb,'pos_arm_joint'+str(j)) for j in range(1,8)]);v=np.array([s.get(rb,'vel_arm_joint'+str(j)) for j in range(1,8)])
  b=np.array([s.get(rb,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']]);a=np.r_[(base-b)/[.87,.87,.994],2.5*(p.ik(z)-q)+.15*v,grip];s,*_=E.step(np.clip(a,E.action_space.low,E.action_space.high))
for z in [.025,.0,-.02,.05,.08]:
 center=p.xyz(s,cube);base=np.r_[center[:2]+[.55,0],np.pi];move(base,.2,0,90)
 for dx in [0,-.025,.025,-.05,.05,-.075,.075]:
  for dy in [0,-.025,.025,-.05,.05]:
   center=p.xyz(s,cube);base=np.r_[center[:2]+[.55+dx,dy],np.pi]
   move(base,z,0,12);before=p.xyz(s,cube).copy();move(base,z,1,3);move(base+[.025,0,0],z,1,3);after=p.xyz(s,cube)
   if np.linalg.norm(after-before)>.01:
    print('MOTION',z,dx,dy,'before',before,'delta',after-before,flush=True)
    move(base+[.025,0,0],.2,1,35);lift=p.xyz(s,cube);print('LIFT',lift,flush=True)
    if lift[2]>.46:E.close();raise SystemExit
   move(base,z,0,2)
 print('DONE Z',z,flush=True)
E.close()

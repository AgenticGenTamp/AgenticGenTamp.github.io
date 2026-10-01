from env_client import make_env
from approach import GeneratedApproach
import numpy as np
E=make_env();s,info=E.reset(seed=0);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,info);rb=p.robot
cubes=[s.get_object_from_name(n) for n in sorted(s.get_object_names()) if n.startswith('cube')]
def move(base,z,grip,n):
 global s
 for _ in range(n):
  q=np.array([s.get(rb,'pos_arm_joint'+str(j)) for j in range(1,8)]);v=np.array([s.get(rb,'vel_arm_joint'+str(j)) for j in range(1,8)]);b=np.array([s.get(rb,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']]);a=np.r_[(base-b)/[.87,.87,.994],2.5*(p.ik(z)-q)+.15*v,grip];s,*_=E.step(np.clip(a,E.action_space.low,E.action_space.high))
for z in [.03,.07,.12,-.02]:
 move(np.array([.75,0,np.pi]),z,0,100)
 for theta in np.arange(0,2*np.pi,np.pi/16):
  base=np.r_[[.55*np.cos(theta),.55*np.sin(theta)],np.pi]
  move(base,z,0,10);before=np.array([p.xyz(s,c) for c in cubes]);move(base,z,1,3);move(base+[.03,0,0],z,1,3);after=np.array([p.xyz(s,c) for c in cubes]);delta=after-before
  if np.max(np.linalg.norm(delta,axis=1))>.008:
   print('MOTION',z,theta,delta.round(3),'BASE',[s.get(rb,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']],'Q',[s.get(rb,'pos_arm_joint'+str(j)) for j in range(1,8)],flush=True);move(base+[.03,0,0],.25,1,40);print('LIFT',np.array([p.xyz(s,c) for c in cubes]).round(3),'BASE',[s.get(rb,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']],'Q',[s.get(rb,'pos_arm_joint'+str(j)) for j in range(1,8)],flush=True)
   if max(p.xyz(s,c)[2] for c in cubes)>.46:E.close();raise SystemExit
 print('DONE',z,flush=True)
E.close()

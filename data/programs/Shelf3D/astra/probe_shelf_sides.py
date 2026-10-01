import numpy as np
from concurrent.futures import ThreadPoolExecutor
from env_client import make_env
from kinematics import HOME
from probe_shelf_drop import pose

def trial(side):
 e=make_env();s,_=e.reset(seed=42)
 r=s.get_objects(e.observation_space.get_type('mujoco_tidybot_robot'))[0]
 obs=s.get_objects(e.observation_space.get_type('mujoco_movable_object'))
 o=min(obs,key=lambda o:s.get(o,'y'));xyz=np.array([s.get(o,f) for f in ['x','y','z']])
 pickup=np.r_[xyz[:2]-[.625,.001],0.]
 if side=='x+':points=[[-.1,-1.2,0],[2.8,-1.2,np.pi],[2.8,0,np.pi],[2.125,.001,np.pi]]
 elif side=='y+':points=[[.3,1.2,0],[1.5,1.2,-np.pi/2],[1.499,.625,-np.pi/2],[1.499,.625,-np.pi/2]]
 else:points=[[.3,-1.2,0],[1.5,-1.2,np.pi/2],[1.501,-.625,np.pi/2],[1.501,-.625,np.pi/2]]
 low=pose(-.04);high=pose(.5-.055);out=[]
 for step in range(440):
  target=HOME if step<35 else (low if step<140 else high)
  base=pickup if step<240 else np.array(points[min(3,(step-240)//40)])
  q=np.array([s.get(r,'pos_arm_joint'+str(i)) for i in range(1,8)])
  b=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']]);d=base-b;d[2]=(d[2]+np.pi)%(2*np.pi)-np.pi
  a=np.r_[np.clip(d,-.1,.1),np.clip((target-q)*2,-.1,.1),float(125<=step<410)]
  s,re,te,tr,info=e.step(a.astype(np.float32))
  if step in [139,239,279,319,359,399,409,439]:out.append((step,np.round([s.get(o,f) for f in ['x','y','z']],4).tolist(),re,te,np.round(b,3).tolist()))
 e.close();print('side',side,out,flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(trial,['x+']))

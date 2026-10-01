from env_client import make_env
import numpy as np
# Scan spare isolated target with original arm pose; base angle pi faces far table.
e=make_env();s,_=e.reset(seed=1);r=s.get_object_from_name('robot');b=s.get_object_from_name('green1')
target=np.array([s.get(b,'pose_x'),s.get(b,'pose_y')])
def pose():return np.array([s.get(r,x) for x in ['base_x','base_y','base_rot']])
def goto(v):
 global s
 for _ in range(100):
  d=np.array(v)-pose();d[2]=(d[2]+np.pi)%(2*np.pi)-np.pi
  if max(abs(d))<1e-5:return True
  a=np.zeros(11);a[:3]=np.clip(d,-.2,.2);a[10]=-1
  p=pose();s,_,t,tr,_=e.step(a)
  if s.get(r,'grasp_active'):
   print('GRASP',pose(),{f:s.get(r,f) for f in e.observation_space.type_features[r.type]},flush=True);return 'grasp'
  if max(abs(p-pose()))<1e-7:return False
 return False
print('target',target,flush=True)
goto([0,-1.5,np.pi]);goto([target[0]+1.1,target[1]-.6,np.pi])
for dx in np.arange(1.1,.25,-.035):
 for dy in (np.arange(-.6,.65,.035) if int(round((1.1-dx)/.035))%2==0 else np.arange(.625,-.625,-.035)):
  z=goto([target[0]+dx,target[1]+dy,np.pi])
  if z=='grasp': e.close();quit()
 print('row',round(dx,3),'p',pose(),flush=True)
e.close()

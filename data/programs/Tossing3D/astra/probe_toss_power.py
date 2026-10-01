import numpy as np
from concurrent.futures import ThreadPoolExecutor
from env_client import make_env
from approach import GeneratedApproach
from kinova_candidate import fk

def trial(spec):
 name,vel,holds=spec;e=make_env()
 try:
  s,info=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);c=p.cube;rob=p.robot;b=s.get_object_from_name('bin_0')
  def xyz(o):return np.array([s.get(o,f) for f in ['x','y','z']])
  def qnow():return np.array([s.get(rob,'pos_arm_joint'+str(j)) for j in range(1,8)])
  def control(q,base=None):
   a=np.zeros(18);err=q-qnow();a[3:10]=np.clip(err,-.1,.1);a[11:]=5*err;a[10]=1
   if base is not None:a[:3]=np.clip(base-np.array([s.get(rob,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']]),-.1,.1)
   return a
  for k in range(140):s,*_=e.step(p.get_action(s))
  qt=p.ik(.4,.85);base=np.array([.9,xyz(b)[1]-.00135,0])
  for k in range(55):s,*_=e.step(control(qt,base))
  initial=xyz(c);q=qnow();jac=np.column_stack([(fk(q+np.eye(7)[j]*1e-4,.12,(0,0,.35))[0]-fk(q,.12,(0,0,.35))[0])/1e-4 for j in range(7)])
  if name.startswith('jac'):
   vel=np.linalg.pinv(jac)@np.array([1.,0,1.]);vel*=6/max(abs(vel))
  vel=np.array(vel);rows=[]
  for k in range(holds+17):
   a=np.zeros(18)
   if k<=holds:a[11:]=vel;a[3:10]=np.clip(vel*.1,-.1,.1)
   a[10]=int(k<holds)
   s,r,d,tr,info=e.step(a)
   if k<=holds+2 or k==holds+16:rows.append([k,*xyz(c).round(4),*[round(s.get(c,f),4) for f in ['vx','vy','vz']]])
  return {'name':name,'holds':holds,'initial':initial.round(3).tolist(),'vel':np.round(vel,3).tolist(),'jac':jac[:,[1,3,5]].round(3).tolist(),'rows':rows}
 finally:e.close()
specs=[('j2',[0,-6,0,0,0,0,0],n) for n in [1,2,3]]+[('jac',None,n) for n in [1,2,3]]
with ThreadPoolExecutor(max_workers=3) as pool:
 for out in pool.map(trial,specs):print(out,flush=True)

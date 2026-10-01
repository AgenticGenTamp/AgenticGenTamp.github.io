from env_client import make_env
import numpy as np

def snap(e,s):
 r=s.get_objects(e.observation_space.get_type('kin_robot'))[0]
 return {f:round(s.get(r,f),3) for f in ('x','y','theta','arm_joint','arm_length','finger_gap')}
def run(name,actions):
 e=make_env(); s,_=e.reset(seed=42)
 print(name,'start',snap(e,s),flush=True)
 for a,n in actions:
  for _ in range(n): s,rew,term,trunc,info=e.step(np.array(a,dtype=float))
  print('after',a,n,snap(e,s),flush=True)
 print('small',[(round(s.get(o,'x'),3),round(s.get(o,'y'),3)) for t in ('small_circle','small_square') for o in s.get_objects(e.observation_space.get_type(t))],flush=True)
 e.close()
run('bounds',[([-.03,0,0,-.08,0],30),([0,-.03,0,0,-.015],30),([.03,0,0,.08,.015],70),([0,.03,0,0,0],70)])
run('base_sweep',[([-.03,-.03,0,-.08,0],30),([.03,0,0,0,0],70)])

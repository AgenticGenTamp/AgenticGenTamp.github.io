from env_client import make_env
import numpy as np

def val(s,r,f): return float(s.get(r,f))
def coords(s,r): return np.array([val(s,r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{i}' for i in range(1,8)]])
env=make_env()
for delta in [.7,.65,.6,.55,.5,.75]:
 for x in [-.17,-.15,-.13,-.11,-.09,-.07]:
  s,_=env.reset(seed=0);r=s.get_object_from_name('robot');home=coords(s,r)
  goal=home.copy();goal[0]=x;goal[1]=.248
  for k in range(2):
   a=np.zeros(11);a[:10]=np.clip(goal-coords(s,r),-.2,.2);s,*_=env.step(a)
  goal[4]+=delta;goal[8]+=delta
  for k in range(8):
   a=np.zeros(11);a[:10]=np.clip(goal-coords(s,r),-.1,.1);a[10]=-1;s,*_=env.step(a)
   if val(s,r,'grasp_active'):
    print('SUCCESS',delta,x,'robot',[(f,val(s,r,f)) for f in env.observation_space.type_features[r.type]],flush=True)
    print('objects',[(o.name,[(f,val(s,o,f)) for f in env.observation_space.type_features[o.type]]) for ty in env.observation_space.types for o in s.get_objects(ty) if o.name!='robot'],flush=True)
    env.close();raise SystemExit
  print('CONFIG',delta,x,np.round(coords(s,r)[[0,1,4,8]],3),flush=True)
print('NO SUCCESS',flush=True);env.close()

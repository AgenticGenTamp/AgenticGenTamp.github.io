from env_client import make_env
import numpy as np

def positions(e,s):
 return {o.name:np.array([s.get(o,a) for a in ['x','y','z']]) for o in s.get_objects(e.observation_space.get_type('mujoco_movable_object'))}
for mode in ['grasp','x','y']:
 e=make_env();s,_=e.reset(seed=0);ro=s.get_object_from_name('robot'); init=positions(e,s)
 print(mode,'initial',init,flush=True)
 cmds=[(0,0,1),(.1,0,1),(.1,0,1),(.1,0,1)] if mode=='grasp' else [(0.1,0,0),(0,0,0),(-.1,0,0),(0,0,0)]
 if mode=='y':cmds=[(y,x,g) for x,y,g in cmds]
 for x,y,g in cmds:
  a=np.zeros(11);a[0]=x;a[1]=y;a[10]=g
  s,r,t,tr,info=e.step(a)
  print('cmd',x,y,g,'base',[round(s.get(ro,f),7) for f in ['pos_base_x','pos_base_y','vel_base_x','vel_base_y']], 'movables',{k:np.round(v-init[k],5).tolist() for k,v in positions(e,s).items()},flush=True)
 e.close()

from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
r = obs.get_object_from_name("robot")
def g(o,f): return round(float(o.get(r,f)),4)
def show(o,tag):
    print(tag, [g(o,f) for f in ['pos_base_x','pos_base_y','pos_base_rot','finger_state','grasp_active']])
show(obs,'init')
for dx,dy,dr in [(-0.2,0,0),(0,0.2,0),(0,0,0.2),(0.05,0,0)]:
    a=np.zeros(11); a[0]=dx;a[1]=dy;a[2]=dr
    obs,rew,term,trunc,info = env.step(a)
    show(obs,f'base {dx},{dy},{dr}')
# gripper repeated close
for i in range(4):
    a=np.zeros(11); a[10]=-1.0
    obs,*_ = env.step(a)
    show(obs,'close')
for i in range(4):
    a=np.zeros(11); a[10]=1.0
    obs,*_ = env.step(a)
    show(obs,'open')
env.close()

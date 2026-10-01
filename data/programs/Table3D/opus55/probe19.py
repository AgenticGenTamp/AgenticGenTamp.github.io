import numpy as np
from env_client import make_env
from approach import *
env=make_env(); obs,info=env.reset(seed=0)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
name=ap.fast[0][0]; acts=ap.fast[1]
for a in acts[:-1]: obs,*_=env.step(a)
r=obs.get_object_from_name('robot'); c=obs.get_object_from_name(name)
print('grasp',obs.get(r,'grasp_active'))
rows=[]
for j,dv in [(None,0),(5,0.2),(5,0.2),(4,0.3),(6,0.3)]:
    if j is not None:
        a=np.zeros(11,dtype=np.float32); a[3+j]=dv; obs,rw,te,tr,_=env.step(a)
    x=ap._x(obs); bx,by,br=x[:3]; T=fk(x[3:]); cb,sb=np.cos(br),np.sin(br)
    Rb=np.array([[cb,-sb,0],[sb,cb,0],[0,0,1]])
    fl=Rb@(T[:3,3]+[ARM_OFFSET_X,0,0])+[bx,by,0]; R=Rb@T[:3,:3]
    cp=np.array([obs.get(c,f) for f in ['pose_x','pose_y','pose_z']])
    print('cube',cp.round(4),'flange(h=0)',fl.round(4),'zaxis',R[:,2].round(3), 'local', (R.T@(cp-fl)).round(4))
env.close()

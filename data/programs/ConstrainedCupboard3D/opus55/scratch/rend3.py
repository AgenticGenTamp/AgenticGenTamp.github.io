import sys; sys.path.insert(0,'.')
from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=11, options={'object_count':1})
R = env.observation_space.get_type('mujoco_tidybot_robot'); M = env.observation_space.get_type('mujoco_movable_object'); F = env.observation_space.get_type('mujoco_fixture')
r = obs.get_objects(R)[0]; s = obs.copy(); s.set(r,'pos_base_x',-3)
dx=float(sys.argv[1]); ang=float(sys.argv[2])
for f in s.get_objects(F):
    print(f.name, round(s.get(f,'y'),2))
    y = s.get(f,'y'); c,sn=np.cos(ang),np.sin(ang)
    s.set(f,'x', dx + (0)*c - y*sn); s.set(f,'y', y*c)
    s.set(f,'y', 0 + y*c); s.set(f,'x', dx - y*sn)
    q = np.array([0.7071,0,0,0.7071]); h=ang/2; qa=np.array([np.cos(h),0,0,np.sin(h)])
    w1,x1,y1,z1=qa; w2,x2,y2,z2=q
    qq=[w1*w2-z1*z2, 0,0, w1*z2+z1*w2]
    for k,v in zip(['qw','qx','qy','qz'],qq): s.set(f,k,v)
for m in s.get_objects(M): s.set(m,'x',-3)
print(env.render_state(state=s, label=sys.argv[3]))
env.close()

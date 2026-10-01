from env_client import make_env
from kin import *
import numpy as np, sys, time
env = make_env()
seed=int(sys.argv[1]); z=float(sys.argv[2]); yaw=float(sys.argv[3])
def q_of(o): return np.array([o.get(R,f'joint_{i}') for i in range(1,8)])
def base_of(o): return (o.get(R,'pos_base_x'),o.get(R,'pos_base_y'),o.get(R,'pos_base_rot'))
obs, info = env.reset(seed=seed)
R = obs.get_object_from_name('robot'); tb=obs.get_object_from_name('target_block')
bx,by = obs.get(tb,"pose_x")-float(__import__("os").environ.get("DX","0")),obs.get(tb,"pose_y")
offs=np.arange(-0.06,0.061,0.02); t0=time.time(); n=0
for dx in offs:
  for dy in offs:
    qt,e = ik(base_of(obs), q_of(obs), np.array([bx+dx,by+dy,z]), down_R(yaw))
    for _ in range(30):
        q=q_of(obs); d=qt-q
        if np.max(np.abs(d))<1e-4: break
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(d,-0.2,0.2)
        obs2,*_=env.step(a); n+=1
        if np.allclose(q_of(obs2),q): break
        obs=obs2
    a=np.zeros(11,dtype=np.float32); a[10]=-1; obs,*_=env.step(a)
    g=obs.get(R,'grasp_active')
    if g:
        print('GRASP', z, dx,dy, [round(obs.get(R,f),4) for f in ['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']]); sys.exit()
    a[10]=1; obs,*_=env.step(a)
print('none at z',z, 'steps',n,'time',time.time()-t0)

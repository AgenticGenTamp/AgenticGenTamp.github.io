from env_client import make_env
from kin import *
import numpy as np, sys
np.set_printoptions(precision=4,suppress=True)
env = make_env()
seed=int(sys.argv[1]); z=float(sys.argv[2]); yaw=float(sys.argv[3]); dx=float(sys.argv[4]); dy=float(sys.argv[5])
def q_of(o): return np.array([o.get(R,f'joint_{i}') for i in range(1,8)])
def base_of(o): return (o.get(R,'pos_base_x'),o.get(R,'pos_base_y'),o.get(R,'pos_base_rot'))
obs, info = env.reset(seed=seed)
R = obs.get_object_from_name('robot'); tb=obs.get_object_from_name('target_block')
bx,by = obs.get(tb,'pose_x'),obs.get(tb,'pose_y')
qt,e = ik(base_of(obs), q_of(obs), np.array([bx+dx,by+dy,z]), down_R(yaw))
for _ in range(30):
    q=q_of(obs); d=qt-q
    if np.max(np.abs(d))<1e-4: break
    a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(d,-0.2,0.2)
    obs2,*_=env.step(a)
    if np.allclose(q_of(obs2),q): print('blocked'); break
    obs=obs2
for i in range(8):
    a=np.zeros(11,dtype=np.float32); a[10]=-1; obs,*_=env.step(a)
    print(i, obs.get(R,'grasp_active'), obs.get(R,'finger_state'), obs.get(tb,'grasp_active'))

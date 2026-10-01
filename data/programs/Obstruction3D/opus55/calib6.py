from env_client import make_env
from kin import *
import numpy as np, sys, json
env = make_env()
def q_of(o): return np.array([o.get(R,f'joint_{i}') for i in range(1,8)])
def base_of(o): return (o.get(R,'pos_base_x'),o.get(R,'pos_base_y'),o.get(R,'pos_base_rot'))
obs, info = env.reset(seed=1)
R = obs.get_object_from_name('robot'); tb=obs.get_object_from_name('target_block')
bx,by = obs.get(tb,"pose_x")-0.1-0.02, obs.get(tb,"pose_y")-0.04
qt,e = ik(base_of(obs), q_of(obs), np.array([bx,by,0.25]), down_R(0))
for _ in range(30):
    q=q_of(obs); d=qt-q
    if np.max(np.abs(d))<1e-4: break
    a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(d,-0.2,0.2); obs,*_=env.step(a)
a=np.zeros(11,dtype=np.float32); a[10]=-1; obs,*_=env.step(a)
assert obs.get(R,'grasp_active')>0.5
GF=['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']
BF=['pose_x','pose_y','pose_z','pose_qx','pose_qy','pose_qz','pose_qw']
data=[]
rng=np.random.default_rng(0)
def rec():
    data.append(dict(base=list(base_of(obs)), q=q_of(obs).tolist(), blk=[obs.get(tb,f) for f in BF], gtf=[obs.get(R,f) for f in GF]))
rec()
# lift first
a=np.zeros(11,dtype=np.float32); a[4]=-0.15; obs,*_=env.step(a); rec()
for i in range(80):
    a=np.zeros(11,dtype=np.float32); a[2:10]=rng.uniform(-0.2,0.2,8)
    if i%3==0: a[2]=0
    q0=q_of(obs); obs,*_=env.step(a)
    if not np.allclose(q_of(obs),q0): rec()
print(len(data), 'grasp still', obs.get(R,'grasp_active'))
json.dump(data, open('calib_data.json','w'))

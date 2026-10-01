from env_client import make_env
from kin import *
import numpy as np, sys
np.set_printoptions(precision=4,suppress=True)
env = make_env()
seed=int(sys.argv[1]); YAW=float(sys.argv[2])
obs, info = env.reset(seed=seed)
R = obs.get_object_from_name('robot')
tb = obs.get_object_from_name('target_block')
def q_of(o): return np.array([o.get(R,f'joint_{i}') for i in range(1,8)])
def base_of(o): return (o.get(R,'pos_base_x'),o.get(R,'pos_base_y'),o.get(R,'pos_base_rot'))
def goto(qt, grip=0.0, maxsteps=40):
    global obs
    for _ in range(maxsteps):
        q = q_of(obs); d = qt-q
        d = (d+np.pi)%(2*np.pi)-np.pi
        if np.max(np.abs(d))<1e-4: return True
        a = np.zeros(11,dtype=np.float32); a[3:10]=np.clip(d,-0.2,0.2); a[10]=grip
        obs2,r,te,tr,_ = env.step(a)
        if np.allclose(q_of(obs2), q): obs=obs2; return False
        obs=obs2
    return False
bx,by,bz = [obs.get(tb,f) for f in ("pose_x","pose_y","pose_z")]
if len(sys.argv)>3: bx,by=float(sys.argv[3]),float(sys.argv[4])
print('block', bx,by,bz, [obs.get(tb,f'half_extent_{c}') for c in 'xyz'])
q = q_of(obs)
for z in np.arange(0.40, 0.05, -0.01):
    qt,e = ik(base_of(obs), q_of(obs), np.array([bx,by,z]), down_R(YAW))
    ok = goto(qt)
    M = fk(base_of(obs), q_of(obs))
    a = np.zeros(11,dtype=np.float32); a[10]=-1
    obs,*_ = env.step(a)
    g = obs.get(R,'grasp_active'); fs=obs.get(R,'finger_state')
    print(f'z={z:.3f} reached={ok} ee={M[:3,3]} grasp={g} finger={fs}')
    if g>0.5:
        print('grasp_tf', [obs.get(R,f) for f in ['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']])
        print('block now', [obs.get(tb,f) for f in ('pose_x','pose_y','pose_z','grasp_active')])
        break
    a[10]=1; obs,*_ = env.step(a)
    if not ok: print('blocked', q_of(obs), qt); break

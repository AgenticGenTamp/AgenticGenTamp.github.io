import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
seed=int(sys.argv[1]); col=sys.argv[2]; zb=float(sys.argv[3]); xin=float(sys.argv[4])
env = make_env(); obs, info = env.reset(seed=seed); S = Sim(env, obs)
FX = env.observation_space.get_type('mujoco_fixture')
cy = {o.name: obs.get(o,'y') for o in obs.get_objects(FX)}[col]
name = sorted(S.rods())[0]
print('pick', pick_rod(S, name), len(S.rew))
yaw = quat_yaw(S.rods()[name]); b=S.base()
# choose rod-axis direction (0 or pi) closest to current ee yaw
rel = (yaw - 0 + np.pi/2) % np.pi - np.pi/2
Rt = Rdown(yaw - rel)  # nearest of 0/pi equivalent
zc = zb + float(sys.argv[5])
bt = np.array([1.2, cy, 0.0])
move_ee_world(S, np.array([1.55, cy, zc+0.08]), Rt, bt=bt, steps=150)
print('carried', np.round(S.rods()[name][:3],3), len(S.rew))
for gx in [1.6, 1.65, 1.7, 1.75, 1.8, 1.84, xin]:
    move_ee_world(S, np.array([gx, cy, zc]), Rt, steps=40, tol=0.003)
r=S.rods()[name]; print('inserted rod',np.round(r[:3],3),'gp',np.round(grasp_point_world(S),3), 'rew', S.rew[-1])
S.goto(grip=0.0, steps=10)
move_ee_world(S, np.array([xin-0.02, cy, zc+0.06]), Rt, steps=30)
move_ee_world(S, np.array([1.6, cy, zc+0.06]), Rt, steps=40)
print('released', np.round(S.rods()[name][:3],3), S.rew[-1])
Rp = np.column_stack([[0,1,0],[0,0,1],[1,0,0]])  # ee z=+x, fingers close vertically
S.goto(grip=1.0, steps=5)
move_ee_world(S, np.array([1.55, cy, zc]), Rp, steps=80)
for gx in np.arange(1.6, 1.87, 0.02):
    move_ee_world(S, np.array([gx, cy, zc]), Rp, steps=25, tol=0.003)
    r=S.rods()[name]; print(' push',round(gx,2),'rod',np.round(r[:3],3),'gp',np.round(grasp_point_world(S),3),'rew',S.rew[-1])
print('rewards seen', sorted(set(S.rew)))
env.close()

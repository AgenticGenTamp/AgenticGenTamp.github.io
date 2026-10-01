import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
seed=1; col='cupboard_0'; zb=0.145
env = make_env(); obs, info = env.reset(seed=seed); S = Sim(env, obs)
FX = env.observation_space.get_type('mujoco_fixture')
cy = {o.name: obs.get(o,'y') for o in obs.get_objects(FX)}[col]
name = sorted(S.rods())[0]
pick_rod(S, name)
yaw = quat_yaw(S.rods()[name]); rel = (yaw + np.pi/2) % np.pi - np.pi/2
Rt = Rdown(yaw - rel); zc = zb + 0.05
move_ee_world(S, np.array([1.55, cy, zc+0.08]), Rt, bt=np.array([1.2, cy, 0.0]), steps=150)
for gx in [1.6, 1.65, 1.7, 1.75, 1.8, 1.84, 1.87]:
    b=S.base(); qt,e1,e2 = ik(S.q(), world_to_arm(b, np.array([gx,cy,zc])), R_world_to_arm(b,Rt), tool=TOOL)
    n=S.goto(qt=qt, steps=40, tol=0.003)
    r=S.rods()[name]
    print(gx, 'ikerr',round(e1,4), 'track', np.round(np.abs(S.q()-qt).max(),3), 'rod', np.round(r[:3],3), 'rodyaw', round(quat_yaw(r),3), 'gp', np.round(grasp_point_world(S),3))
print(env.render_state(state=S.obs, label='ins3'))
env.close()

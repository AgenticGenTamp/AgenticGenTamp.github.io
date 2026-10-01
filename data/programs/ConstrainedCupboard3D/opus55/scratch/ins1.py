import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
seed=int(sys.argv[1]); col=sys.argv[2]; zc=float(sys.argv[3])
env = make_env(); obs, info = env.reset(seed=seed); S = Sim(env, obs)
FX = env.observation_space.get_type('mujoco_fixture')
cy = {o.name: obs.get(o,'y') for o in obs.get_objects(FX)}[col]
name = sorted(S.rods())[0]
rod = S.rods()[name]; yaw = quat_yaw(rod); u = np.array([-np.sin(yaw), np.cos(yaw), 0])
gp = rod[:3] + 0.12*u
b = S.base(); d = gp[:2]-b[:2]; th=np.arctan2(d[1],d[0])
S.goto(bt=np.array([gp[0]-0.55*np.cos(th), gp[1]-0.55*np.sin(th), th]), steps=80)
Rw = Rdown(yaw)
move_ee_world(S, np.array([gp[0],gp[1],0.15]), Rw, grip=0.0)
move_ee_world(S, np.array([gp[0],gp[1],0.02]), Rw, grip=0.0, steps=60)
S.goto(grip=1.0, steps=8)
move_ee_world(S, np.array([gp[0],gp[1],0.3]), Rw, steps=60)
print('lifted', np.round(S.rods()[name][:3],3))
# carry: rod pointing +x means ee x = -x world  (Rdown(pi))
Rt = Rdown(np.pi)
bt = np.array([1.12, cy, 0.0])
move_ee_world(S, np.array([1.6, cy, zc+0.1]), Rt, bt=bt, steps=150)
print('carried', np.round(S.rods()[name][:3],3), 'gp', np.round(grasp_point_world(S),3), len(S.rew))
for gx in [1.65, 1.7, 1.75, 1.8, 1.84]:
    move_ee_world(S, np.array([gx, cy, zc+0.005]), Rt, steps=40, tol=0.003)
    r=S.rods()[name]; print(' gx',gx,'rod',np.round(r[:3],3),'gp',np.round(grasp_point_world(S),3), 'rew', S.rew[-1])
S.goto(grip=0.0, steps=10)
move_ee_world(S, np.array([1.7, cy, zc+0.1]), Rt, steps=40)
print('released', np.round(S.rods()[name][:3],3), S.rew[-1])
# push with closed fingertips: gripper pointing +x, fingers closing vertically
Rp = np.array([[0,0,1],[1,0,0],[0,1,0.]]).T  # columns: ee x=world y?, set below
Rp = np.column_stack([[0,1,0],[0,0,1],[1,0,0]])  # ee x=y, ee y=z, ee z=+x
S.goto(grip=1.0, steps=5)
for gx in [1.7, 1.75, 1.8, 1.85, 1.9]:
    move_ee_world(S, np.array([gx-0.15+0.12, cy, zc]), Rp, steps=40, tol=0.003)  # tip ~ grasp point+?
    r=S.rods()[name]; print(' push',gx,'rod',np.round(r[:3],3),'gp',np.round(grasp_point_world(S),3),'rew',S.rew[-1])
print('rewards seen', sorted(set(S.rew)))
env.close()

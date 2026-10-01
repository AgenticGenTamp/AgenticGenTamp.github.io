import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
seed=int(sys.argv[1]); col=sys.argv[2]; zb=float(sys.argv[3]); off=float(sys.argv[4]); dz=float(sys.argv[5])
env = make_env(); obs, info = env.reset(seed=seed); S = Sim(env, obs)
FX = env.observation_space.get_type('mujoco_fixture')
cy = {o.name: obs.get(o,'y') for o in obs.get_objects(FX)}[col]
name = sorted(S.rods())[0]
rod = S.rods()[name]; yaw = quat_yaw(rod); u = np.array([-np.sin(yaw), np.cos(yaw), 0])
gp = rod[:3] + off*u
b = S.base(); d = gp[:2]-b[:2]; th=np.arctan2(d[1],d[0])
S.goto(bt=np.array([gp[0]-0.55*np.cos(th), gp[1]-0.55*np.sin(th), th]), steps=80)
Rw = Rdown(yaw)
move_ee_world(S, np.array([gp[0],gp[1],0.15]), Rw, grip=0.0)
move_ee_world(S, np.array([gp[0],gp[1],0.02]), Rw, grip=0.0, steps=60)
S.goto(grip=1.0, steps=8)
move_ee_world(S, np.array([gp[0],gp[1],0.25]), Rw, steps=60)
v = S.rods()[name][:3]-grasp_point_world(S); print('lifted', np.round(S.rods()[name][:3],3), 'center-gp', np.round(v,3))
# want v to point +x: ee x currently along yaw (Rdown(yaw)); v = s*eex; s=sign
s = np.sign(np.dot(v[:2], [np.sin(yaw), -np.cos(yaw)]))
Rt = Rdown(np.pi/2 if s>0 else -np.pi/2)
zc = zb + dz
move_ee_world(S, np.array([1.5, cy, zc+0.08]), Rt, bt=np.array([1.15, cy, 0.0]), steps=150)
print('carried', np.round(S.rods()[name][:3],3), 'gp', np.round(grasp_point_world(S),3), len(S.rew))
for gx in [1.55, 1.6, 1.65, 1.7, 1.75, 1.8]:
    move_ee_world(S, np.array([gx, cy, zc]), Rt, steps=40, tol=0.003)
r=S.rods()[name]; print('inserted rod',np.round(r[:3],3),'gp',np.round(grasp_point_world(S),3))
S.goto(grip=0.0, steps=10)
move_ee_world(S, np.array([1.72, cy, zc+0.06]), Rt, steps=30)
move_ee_world(S, np.array([1.55, cy, zc+0.06]), Rt, steps=40)
r=S.rods()[name]; pz=r[2]; print('released', np.round(S.rods()[name][:3],3), round(quat_yaw(S.rods()[name]),3))
Rp = np.column_stack([[0,0,1],[0,-1,0],[1,0,0]])
S.goto(grip=1.0, steps=5)
move_ee_world(S, np.array([1.5, cy, zc+0.1]), Rp, steps=100); move_ee_world(S, np.array([1.5, cy, pz]), Rp, steps=60)
for gx in np.arange(1.5, 1.72, 0.02):
    move_ee_world(S, np.array([gx, cy, pz]), Rp, steps=25, tol=0.003)
    r=S.rods()[name]; print(' push',round(gx,2),'rod',np.round(r[:3],3),'gp',np.round(grasp_point_world(S),3),'rew',S.rew[-1])
print('rewards seen', sorted(set(S.rew)))
env.close()

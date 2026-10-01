import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
seed=int(sys.argv[1]); col=sys.argv[2]; zb=float(sys.argv[3]); REACH=0.65
env = make_env(); obs, info = env.reset(seed=seed); S = Sim(env, obs)
FX = env.observation_space.get_type('mujoco_fixture')
cy = {o.name: obs.get(o,'y') for o in obs.get_objects(FX)}[col]
names = sorted(S.rods()); A, B = names[0], names[1]
def minrot_to_vertical(R):
    y = R[:, 1]; t = np.array([0, 0, 1.0]) * np.sign(y[2] if abs(y[2])>1e-6 else 1)
    v = np.cross(y, t); s = np.linalg.norm(v); c = np.dot(y, t)
    if s < 1e-9: return R
    K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    Rm = np.eye(3) + K + K@K*((1-c)/s**2)
    return Rm @ R
pick_rod(S, A)
Rv = np.column_stack([[0,1,0],[0,0,1],[1,0,0]])
zc = zb + 0.15 + 0.015; gx0 = 1.6
move_ee_world(S, np.array([gx0, cy, zc]), Rv, bt=np.array([gx0-MX-REACH, cy, 0.0]), steps=200)
for it in range(3):
    G = grasp_tf(S, A); r = S.rods()[A]
    Rrod = minrot_to_vertical(quat_to_R(r[3:7]))
    pe, Re = ee_for_rod(G, np.array([gx0, cy, zc]), Rrod)
    move_ee_world(S, pe, Re, steps=80, tol=0.003)
    r = S.rods()[A]; print('A corrected', np.round(r[:3],3), 'tilt', round(np.degrees(np.arccos(abs(quat_to_R(r[3:7])[2,1]))),1))
qhold = S.q().copy(); b0 = S.base().copy()
for dx in np.arange(0.02, 0.4, 0.02):
    S.goto(qt=qhold, bt=b0+np.array([dx,0,0]), steps=20, tol=0.003)
    r=S.rods()[A]; g=grasp_point_world(S)
    if g[0] < gx0+dx-0.03 - (gx0 - pe[0])*0: 
        pass
    if S.base()[0] < b0[0]+dx-0.02: break
r=S.rods()[A]; print('A inserted', np.round(r[:3],3), 'base', np.round(S.base(),3), 'tilt', round(np.degrees(np.arccos(abs(quat_to_R(r[3:7])[2,1]))),1))
S.goto(grip=0.0, steps=20)
S.goto(bt=S.base()-np.array([0.02,0,0]), steps=20)
S.goto(bt=S.base()-np.array([0.25,0,0]), steps=40)
r=S.rods()[A]; print('A released', np.round(r[:3],3), 'tilt', round(np.degrees(np.arccos(abs(quat_to_R(r[3:7])[2,1]))),1), 'steps', len(S.rew))
# Phase B: pusher rod
pick_rod(S, B)
yaw = quat_yaw(S.rods()[B]); v = S.rods()[B][:3]-grasp_point_world(S)
Rt = Rdown(np.pi/2)
zp = zb + 0.03
move_ee_world(S, np.array([1.45, cy, zp+0.1]), Rt, bt=np.array([1.45-MX-0.5, cy, 0.0]), steps=200)
move_ee_world(S, np.array([1.45, cy, zp]), Rt, steps=60)
print('B ready', np.round(S.rods()[B][:3],3), 'steps', len(S.rew))
qhold = S.q().copy(); b0 = S.base().copy()
for dx in np.arange(0.01, 0.45, 0.01):
    S.goto(qt=qhold, bt=b0+np.array([dx,0,0]), steps=12, tol=0.003)
    ra=S.rods()[A]; rb=S.rods()[B]
    print(' push dx',round(dx,2),'A',np.round(ra[:3],3),'tilt', round(np.degrees(np.arccos(abs(quat_to_R(ra[3:7])[2,1]))),1),'B',np.round(rb[:3],3),'rew',S.rew[-1])
    if S.base()[0] < b0[0]+dx-0.02: break
print('rewards seen', sorted(set(S.rew)), len(S.rew))
env.close()

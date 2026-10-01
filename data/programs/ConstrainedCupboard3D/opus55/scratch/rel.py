import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
seed=11; col='cupboard_0'; zb=0.145; REACH=0.65; mode=sys.argv[1]
env = make_env(); obs, info = env.reset(seed=seed, options={'object_count':1}); S = Sim(env, obs)
FX = env.observation_space.get_type('mujoco_fixture')
cy = {o.name: obs.get(o,'y') for o in obs.get_objects(FX)}[col]
A = sorted(S.rods())[0]
def tilt(r): return round(np.degrees(np.arccos(abs(quat_to_R(r[3:7])[2,1]))),1)
def vert(R):
    y = R[:, 1]; t = np.array([0, 0, np.sign(y[2])]); v = np.cross(y, t); s = np.linalg.norm(v); c = np.dot(y, t)
    K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]); return (np.eye(3) + K + K@K*((1-c)/max(s,1e-9)**2)) @ R
pick_rod(S, A)
Rv = np.column_stack([[0,1,0],[0,0,1],[1,0,0]])
zc = zb + 0.15 + 0.015; gx0 = 1.6
move_ee_world(S, np.array([gx0, cy, zc]), Rv, bt=np.array([gx0-MX-REACH, cy, 0.0]), steps=200)
for it in range(2):
    G = grasp_tf(S, A); pe, Re = ee_for_rod(G, np.array([gx0, cy, zc]), vert(quat_to_R(S.rods()[A][3:7])))
    move_ee_world(S, pe, Re, steps=80, tol=0.003)
qhold = S.q().copy(); b0 = S.base().copy()
xs = float(sys.argv[2])
S.goto(qt=qhold, bt=b0+np.array([xs,0,0]), steps=60, tol=0.003)
r=S.rods()[A]; print('A inserted', np.round(r[:3],3), 'base', np.round(S.base(),3), 'tilt', tilt(r))
for k in range(4):
    S.goto(grip=0.0, steps=5); r=S.rods()[A]; print(' open', k, np.round(r[:3],3), tilt(r))
if mode=='up':
    pe,Re = ee_world(S); move_ee_world(S, pe+np.array([0,0,0.03]), Re, steps=20)
    r=S.rods()[A]; print(' up', np.round(r[:3],3), tilt(r))
for k in range(10):
    S.goto(bt=S.base()-np.array([0.01*(k+1),0,0]), steps=4)
    r=S.rods()[A]; print(' back', k, np.round(r[:3],3), tilt(r), 'gp', np.round(grasp_point_world(S),3))
print('rewards', sorted(set(S.rew)), len(S.rew)); env.close()

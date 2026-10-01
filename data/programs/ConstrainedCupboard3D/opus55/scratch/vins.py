import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
seed=int(sys.argv[1]); col=sys.argv[2]; zb=float(sys.argv[3]); side=int(sys.argv[4]); REACH=float(sys.argv[5])
env = make_env(); obs, info = env.reset(seed=seed); S = Sim(env, obs)
FX = env.observation_space.get_type('mujoco_fixture')
cy = {o.name: obs.get(o,'y') for o in obs.get_objects(FX)}[col]
name = sorted(S.rods())[0]
print('pick', pick_rod(S, name), len(S.rew))
if side>0: Rv = np.column_stack([[0,1,0],[0,0,1],[1,0,0]])
else: Rv = np.column_stack([[0,-1,0],[0,0,-1],[1,0,0]])
zc = zb + 0.15 + 0.02
gx0 = 1.5
bt = np.array([gx0-MX-REACH, cy, 0.0])
e,n=move_ee_world(S, np.array([gx0, cy, zc]), Rv, bt=bt, steps=200)
r=S.rods()[name]; print('carried ikerr',round(e,4), n, np.round(r,3), 'gp', np.round(grasp_point_world(S),3), len(S.rew))
qhold = S.q().copy()
for gx in np.arange(1.52, 1.95, 0.02):
    S.goto(qt=qhold, bt=np.array([gx-MX-REACH, cy, 0.0]), steps=20, tol=0.003)
    r=S.rods()[name]; g=grasp_point_world(S)
    print(' ins',round(gx,2),'rod',np.round(r[:3],3),'gp',np.round(g,3),'base',np.round(S.base(),3),'rew',S.rew[-1])
    if g[0] < gx-0.03: break
S.goto(grip=0.0, steps=15)
r=S.rods()[name]; print('released', np.round(r,3), 'rew', S.rew[-1])
S.goto(bt=S.base()-np.array([0.15,0,0]), steps=30)
r=S.rods()[name]; print('retreated', np.round(r,3), 'rew', S.rew[-5:])
print('rewards seen', sorted(set(S.rew)), len(S.rew))
env.close()

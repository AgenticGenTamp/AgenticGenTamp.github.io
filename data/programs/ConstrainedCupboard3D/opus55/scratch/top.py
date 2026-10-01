import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
seed=int(sys.argv[1]); col=sys.argv[2]; oc=int(sys.argv[3]); ztop=float(sys.argv[4])
env = make_env(); obs, info = env.reset(seed=seed, options={'object_count':oc}); S = Sim(env, obs)
FX = env.observation_space.get_type('mujoco_fixture')
cols = {o.name: (obs.get(o,'x'),obs.get(o,'y')) for o in obs.get_objects(FX)}
cx, cy = cols[col]
A = sorted(S.rods())[0]
pick_rod(S, A)
Rw = Rdown(np.pi/2)
move_ee_world(S, np.array([cx, cy, ztop+0.2]), Rw, bt=np.array([cx-0.62, cy, 0.0]), steps=250)
r=S.rods()[A]; print('above', np.round(r[:3],3), len(S.rew))
for it in range(2):
    G=grasp_tf(S,A); pe,Re=ee_for_rod(G, np.array([cx,cy,ztop+0.035]), np.array([[0,1.,0],[-1,0,0],[0,0,1]]) if quat_to_R(S.rods()[A][3:7])[0,1]<0 else np.array([[0,-1.,0],[1,0,0],[0,0,1]]))
    move_ee_world(S, pe, Re, steps=80, tol=0.003)
r=S.rods()[A]; print('lowered', np.round(r[:3],3), 'gp', np.round(grasp_point_world(S),3))
S.goto(grip=0.0, steps=15)
move_ee_world(S, np.array([cx, cy, ztop+0.2]), Rw, steps=60)
S.goto(steps=30)
r=S.rods()[A]; print('final', np.round(r,3), 'term', S.term, len(S.rew))

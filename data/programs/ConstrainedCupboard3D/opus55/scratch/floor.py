import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
seed=int(sys.argv[1]); col=sys.argv[2]; pz=float(sys.argv[3]); REACH=0.65
env = make_env(); obs, info = env.reset(seed=seed, options={'object_count':1}); S = Sim(env, obs)
FX = env.observation_space.get_type('mujoco_fixture')
cols = {o.name: (obs.get(o,'x'),obs.get(o,'y'),obs.get(o,'z')) for o in obs.get_objects(FX)}
cy = cols[col][1]; print('target', cols[col])
A = sorted(S.rods())[0]
term=[False]; _st=S.step
def st(a):
    r=_st(a)
    return r
pick_rod(S, A)
print('picked', np.round(S.rods()[A][:3],3), 'rews', set(S.rew))
# place on floor along x, center at 1.62
Rw = Rdown(np.pi/2)
move_ee_world(S, np.array([1.62, cy, 0.15]), Rw, bt=np.array([1.62-0.55, cy, 0.0]), steps=200)
G = grasp_tf(S, A)
move_ee_world(S, np.array([1.62, cy, 0.035]), Rw, steps=80)
S.goto(grip=0.0, steps=10)
move_ee_world(S, np.array([1.62, cy, 0.15]), Rw, steps=40)
r=S.rods()[A]; print('placed', np.round(r,3))
# push with narrow pose
ar=np.radians(float(sys.argv[4])); zz=np.array([np.cos(ar),0,-np.sin(ar)]); xx=np.array([0,1.,0]); Rp=np.column_stack([xx,np.cross(zz,xx),zz]); REACH=0.6
S.goto(grip=1.0, steps=3)
xr = r[0]-0.15-0.03
e,n = move_ee_world(S, np.array([xr, cy, pz+0.1]), Rp, bt=np.array([xr-MX-REACH, cy, 0.0]), steps=200)
e,n = move_ee_world(S, np.array([xr, cy, pz]), Rp, steps=80, tol=0.003)
print('push pose ok' , e, np.round(grasp_point_world(S),3))
qhold=S.q().copy(); b0=S.base().copy()
for k in range(1,50):
    S.goto(qt=qhold, bt=b0+np.array([0.01*k,0,0]), steps=6, tol=0.003)
    r=S.rods()[A]
    if k%5==0: print(k, 'rod', np.round(r[:3],3), 'gp', np.round(grasp_point_world(S),3), 'rews', sorted(set(S.rew))[-2:], 'n', len(S.rew))
    if S.base()[0] < b0[0]+0.01*k-0.03: print('blocked'); break
print('final rod', np.round(S.rods()[A],3), 'steps', len(S.rew), 'maxrew', max(S.rew))

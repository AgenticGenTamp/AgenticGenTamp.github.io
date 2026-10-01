import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
seed=int(sys.argv[1]); colidx=int(sys.argv[2]); zb=float(sys.argv[3]); off=float(sys.argv[4]); oc=int(sys.argv[5])
env = make_env(); obs, info = env.reset(seed=seed, options={'object_count':oc}); S = Sim(env, obs)
FX = env.observation_space.get_type('mujoco_fixture')
cy = {o.name: obs.get(o,'y') for o in obs.get_objects(FX)}['cupboard_%d'%colidx]
A = sorted(S.rods())[0]
term=[False]
_st = S.step
def st(a):
    t=_st(a); 
    if t: term[0]=True; print('TERMINATED! rod', np.round(S.rods()[A][:3],3), 'rew', S.rew[-1])
    return t
S.step = st
rod = S.rods()[A]; yaw = quat_yaw(rod); u = np.array([-np.sin(yaw), np.cos(yaw), 0])
gp = rod[:3] - off*u
b = S.base(); d = gp[:2]-b[:2]; th=np.arctan2(d[1],d[0])
S.goto(bt=np.array([gp[0]-0.55*np.cos(th), gp[1]-0.55*np.sin(th), th]), steps=80)
Rw = Rdown(yaw)
move_ee_world(S, np.array([gp[0],gp[1],0.15]), Rw, grip=0.0)
move_ee_world(S, np.array([gp[0],gp[1],0.02]), Rw, grip=0.0, steps=60)
S.goto(grip=1.0, steps=8)
move_ee_world(S, np.array([gp[0],gp[1],0.25]), Rw, steps=60)
# desired rod orientation: local y along world x (either sign), level
def horiz(R):
    y = R[:,1]; s = np.sign(y[0]) if abs(y[0])>1e-6 else 1.0
    ny = np.array([s,0,0]); nz = np.array([0,0,1.0]); nx = np.cross(ny, nz)
    return np.column_stack([nx, ny, nz])
zr = zb + float(sys.argv[6])   # rod center height during insertion
G = grasp_tf(S, A)
def rod_goal(xc, z):
    Rr = horiz(quat_to_R(S.rods()[A][3:7]))
    return ee_for_rod(grasp_tf(S, A), np.array([xc, cy, z]), Rr)
# ee must be behind rod center: rod center = ee + offset along +x
pe, Re = rod_goal(1.5, zr+0.08)
print('ee x offset from rod', round(1.5-pe[0],3))
bt = np.array([pe[0]-MX-0.55, cy, 0.0])
move_ee_world(S, pe, Re, bt=bt, steps=200)
for it in range(2):
    pe, Re = rod_goal(1.5, zr+0.08); move_ee_world(S, pe, Re, steps=60, tol=0.003)
pe, Re = rod_goal(1.5, zr); move_ee_world(S, pe, Re, steps=60, tol=0.003)
r=S.rods()[A]; print('ready', np.round(r[:3],3), 'len', len(S.rew))
qhold = S.q().copy(); b0 = S.base().copy()
for dx in np.arange(0.02, 0.6, 0.02):
    S.goto(qt=qhold, bt=b0+np.array([dx,0,0]), steps=15, tol=0.003)
    if S.base()[0] < b0[0]+dx-0.015: break
r=S.rods()[A]; print('inserted', np.round(r[:3],3), 'gp', np.round(grasp_point_world(S),3))
S.goto(grip=0.0, steps=15)
r=S.rods()[A]; print('released', np.round(r[:3],3))
pe,Re = ee_world(S); move_ee_world(S, pe+np.array([0,0,0.08]), Re, steps=30)
S.goto(bt=S.base()-np.array([0.3,0,0]), steps=40)
r=S.rods()[A]; print('retreated', np.round(r[:3],3), 'len', len(S.rew))
# push: ee z=+x, fingers closing along z (thin across y)
ar=np.radians(45); zz=np.array([np.cos(ar),0,-np.sin(ar)]); xx=np.array([0,1.,0]); Rp=np.column_stack([xx,np.cross(zz,xx),zz])
S.goto(grip=1.0, steps=3)
xr = r[0]-0.15
b = S.base()
e,_ = move_ee_world(S, np.array([xr-0.08, cy, r[2]+0.05]), Rp, bt=np.array([xr-0.08-MX-0.6, cy, 0]), steps=200)
move_ee_world(S, np.array([xr-0.05, cy, r[2]+0.01]), Rp, steps=60, tol=0.003)  # tool tip ~ gp+?
print('push ready gp', np.round(grasp_point_world(S),3), 'rod', np.round(S.rods()[A][:3],3), len(S.rew))
qhold = S.q().copy(); b0 = S.base().copy()
for dx in np.arange(0.01, 0.4, 0.01):
    S.goto(qt=qhold, bt=b0+np.array([dx,0,0]), steps=8, tol=0.003)
    r=S.rods()[A]
    if int(dx*100)%3==0: print(' push', round(dx,2), 'rod', np.round(r[:3],3), 'gp', np.round(grasp_point_world(S),3))
    if S.base()[0] < b0[0]+dx-0.015 or term[0]: break
print('final rod', np.round(S.rods()[A],3), 'rewards', sorted(set(S.rew)), len(S.rew))
env.close()

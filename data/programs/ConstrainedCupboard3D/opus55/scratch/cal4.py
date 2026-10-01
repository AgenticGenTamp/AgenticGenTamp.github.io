import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
from scipy.optimize import least_squares
MX, MZ, TOOL = 0.12, 0.3948, 0.125
def quat_yaw(qv): w,x,y,z = qv[3:7]; return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
def Rdown(yaw):
    c,s=np.cos(yaw),np.sin(yaw); return np.array([[c,s,0],[s,-c,0],[0,0,-1.]])
def to_arm(S, pw):
    b=S.base(); c,s=np.cos(b[2]),np.sin(b[2]); d=pw[:2]-b[:2]
    return np.array([c*d[0]+s*d[1]-MX, -s*d[0]+c*d[1], pw[2]-MZ])
env = make_env(); obs, info = env.reset(seed=1)
S = Sim(env, obs)
S.goto(bt=np.array([0.0,-0.5,0.0]), steps=40); S.goto(bt=np.array([0.0,-0.339,0.0]), steps=40)
rod = S.rods()['cuboid_0']; gya = quat_yaw(rod)-S.base()[2]
for z,g in [(0.15,0.0),(0.015,0.0),(0.015,1.0),(0.25,1.0)]:
    qt,e1,_ = ik(S.q(), to_arm(S, np.array([rod[0],rod[1],z])), Rdown(gya), tool=TOOL); S.goto(qt=qt, grip=g, steps=60)
data=[]
rng=np.random.default_rng(0)
for k in range(14):
    p = np.array([rng.uniform(0.35,0.6), rng.uniform(-0.25,0.25), rng.uniform(-0.25,0.1)])
    yaw = rng.uniform(-1,1)
    R = Rdown(yaw)
    if k%2: 
        R = R @ np.array([[np.cos(0.4),0,np.sin(0.4)],[0,1,0],[-np.sin(0.4),0,np.cos(0.4)]])
    bt = S.base() + np.array([rng.uniform(-0.1,0.1), rng.uniform(-0.1,0.1), rng.uniform(-0.3,0.3)])
    qt,e1,_ = ik(S.q(), p, R, tool=TOOL)
    S.goto(qt=qt, bt=bt, steps=80, tol=0.004)
    for _ in range(5): S.goto(qt=qt, bt=bt, steps=1)
    data.append((S.q(), S.base(), S.rods()['cuboid_0']))
print('last rod z', data[-1][2][2])
np.save('scratch/caldata.npy', np.array([np.concatenate(d) for d in data]))
def resid(x):
    mx,my,mz,ox,oy,oz = x; out=[]
    for q,b,rd in data:
        T = fk(q, 0.0); pa = T[:3,3] + T[:3,:3]@np.array([ox,oy,oz]) + np.array([mx,my,mz])
        c,s=np.cos(b[2]),np.sin(b[2])
        pw = np.array([b[0]+c*pa[0]-s*pa[1], b[1]+s*pa[0]+c*pa[1], pa[2]])
        out.append(pw-rd[:3])
    return np.concatenate(out)
sol = least_squares(resid, [0.12,0,0.39,0,0,0.13])
print(np.round(sol.x,4), 'rms', np.sqrt(np.mean(sol.fun**2)))
print(np.round(sol.fun.reshape(-1,3),3))
env.close()

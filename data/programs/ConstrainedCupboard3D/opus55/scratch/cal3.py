import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
MX, MZ, TOOL = 0.12, 0.3948, 0.125
def quat_yaw(qv): w,x,y,z = qv[3:7]; return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
def Rdown(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,s,0],[s,-c,0],[0,0,-1.]])  # z down, x at yaw
def to_arm(S, pw):
    b=S.base(); c,s=np.cos(b[2]),np.sin(b[2])
    d=pw[:2]-b[:2]; return np.array([c*d[0]+s*d[1]-MX, -s*d[0]+c*d[1], pw[2]-MZ])
for variant in [0, 1]:
    env = make_env(); obs, info = env.reset(seed=1)
    S = Sim(env, obs)
    S.goto(bt=np.array([0.0,-0.5,0.0]), steps=40); S.goto(bt=np.array([0.0,-0.339,0.0]), steps=40)
    rod = S.rods()['cuboid_0']; ry = quat_yaw(rod)
    gy = ry + (np.pi/2 if variant==0 else 0.0)   # ee x-axis direction
    b=S.base(); gya = gy - b[2]
    for z,g in [(0.15,0.0),(0.015,0.0),(0.015,1.0),(0.25,1.0)]:
        pa = to_arm(S, np.array([rod[0],rod[1],z]))
        qt,e1,_ = ik(S.q(), pa, Rdown(gya), tool=TOOL); n=S.goto(qt=qt, grip=g, steps=60)
        print(variant, z, g, 'steps',n,'grip',round(S.obs.get(S.r,'pos_gripper'),3), 'rod', np.round(S.rods()['cuboid_0'][:3],3))
    env.close()

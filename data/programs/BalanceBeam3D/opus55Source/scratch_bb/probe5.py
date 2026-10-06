import time, numpy as np, mujoco
np.set_printoptions(precision=4, suppress=True)
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env=KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
ke=env._kinder_env; oc=ke._object_centric_env; re=oc._robot_env
ss=oc._objects_dict['seesaw_1']
# closed-loop absolute arm tracking
obs,_=env.reset(seed=0)
q0=obs[19:26].copy(); qt=q0+np.array([0.5,0.3,-0.4,0.5,0.3,-0.3,0.4])
for k in range(40):
    o=env.get_state(); a=np.zeros(11,np.float32); a[3:10]=np.clip(qt-o[19:26],-0.1,0.1)
    o,*_=env.step(a)
    if k%5==4: print("k",k+1,"max |err|",np.abs(qt-o[19:26]).max().round(4))
# also unclipped absolute delta (one-shot full error) to see
obs,_=env.reset(seed=0); qt=obs[19:26]+0.5
for k in range(20):
    o=env.get_state(); a=np.zeros(11,np.float32); a[3:10]=qt-o[19:26]
    o,*_=env.step(a)
    if k%4==3: print(" unclipped k",k+1,"max|err|",np.abs(qt-o[19:26]).max().round(4))
# balance tests via teleport
def test(offs, label):
    obs,_=env.reset(seed=0)
    sx,sy,sz=obs[38:41]
    s=obs.copy()
    top=sz+0.04+0.0025+0.005
    for (i,half),off in zip([(0,0.018),(54,0.012),(70,0.012)],offs):
        s[i:i+3]=[sx, sy+off, top+half+0.001]; s[i+3:i+7]=[0.7071,0,0,0.7071]; s[i+7:i+13]=0
    oc._set_state(ke.observation_space.devectorize(s))
    for k in range(30): o,r,term,tr,_=env.step(np.zeros(11,np.float32))
    print(label, "offs",offs,"tilt %.2f deg"%ss.get_beam_tilt_angle_degrees(),"term",term,"r",r, "LB",o[0:3],"S1",o[54:57])
test([0,-0.06,0.06],"sym")
test([0,0.06,0.12],"LB center, smalls one side")
test([-0.06,0.06,0.06],"LB -0.06, smalls +0.06 (balanced moments)")
test([-0.1,0.1,0.1],"LB -0.1 smalls +0.1")
test([0.05,0,0],"LB +0.05 smalls center?") 
test([0.1,-0.03,0.03],"LB +0.1")

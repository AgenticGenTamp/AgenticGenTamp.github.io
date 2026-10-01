from env_client import make_env
import numpy as np, kin, sys
YAW=float(sys.argv[1]); ZG=float(sys.argv[2]); WALLY=float(sys.argv[3]) if len(sys.argv)>3 else -0.35
env = make_env()
obs, info = env.reset(seed=3, options={'object_count':10})
R = obs.get_object_from_name('robot'); YB=obs.get_object_from_name('bin_yellow_0')
def base(): return obs.data[R][:3].copy()
def qq(): return obs.data[R][3:10].copy()
def yb(): return obs.data[YB][:7].copy()
TOOL=0.12
G=[0.0]; QD=[None]
sc=[0]
def step(bc=(0,0,0)):
    global obs
    a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(bc,-0.1,0.1); a[10]=G[0]
    if QD[0] is not None: a[3:10]=np.clip(QD[0]-qq(),-0.1,0.1)
    obs,rew,term,trunc,info=env.step(a); sc[0]+=1
    return rew,term
def arm_to(tgt,yaw=YAW,n=80,tol=0.01,bc=(0,0,0)):
    qd,err=kin.ik(np.array(tgt), kin.rot_down(yaw), qq(), TOOL); QD[0]=qd
    for k in range(n):
        step(bc)
        if np.abs(qd-qq()).max()<tol: break
    return err
def base_to(bx,by,bth=0.0,n=40,tol=0.01):
    for k in range(n):
        b=base(); e=np.array([bx-b[0],by-b[1],bth-b[2]])
        if np.abs(e).max()<tol: break
        step(np.clip(e,-0.1,0.1))
G[0]=0.0
base_to(0.5, WALLY, 0.0)
b=base(); fx=0.5-b[0]; fy=WALLY-b[1]
print('base',np.round(b,3),'fx',round(fx,3),'yellow',np.round(yb(),3))
print('ikerr',round(arm_to((fx,fy,0.36)),4),'steps',sc[0])
print('ikerr',round(arm_to((fx,fy,ZG)),4),'steps',sc[0],'yellow',np.round(yb(),3))
G[0]=1.0
for i in range(12): step()
print('after close, yellow',np.round(yb(),3))
arm_to((fx,fy,ZG+0.18))
print('after lift, yellow',np.round(yb(),4),'gripper',round(obs.data[R][10],3),'steps',sc[0])
env.close()

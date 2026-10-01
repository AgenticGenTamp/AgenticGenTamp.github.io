from env_client import make_env
import numpy as np, kin, sys
YAW=float(sys.argv[1]); WALLY=float(sys.argv[2]); WX=float(sys.argv[3])
env = make_env()
obs, info = env.reset(seed=3, options={'object_count':10})
R = obs.get_object_from_name('robot'); YB=obs.get_object_from_name('bin_yellow_0')
def base(): return obs.data[R][:3].copy()
def qq(): return obs.data[R][3:10].copy()
def yb(): return obs.data[YB][:7].copy()
TOOL=0.12; G=[0.0]; QD=[None]; sc=[0]
def step(bc=(0,0,0)):
    global obs
    a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(bc,-0.1,0.1); a[10]=G[0]
    if QD[0] is not None: a[3:10]=np.clip(QD[0]-qq(),-0.1,0.1)
    obs,rew,term,trunc,info=env.step(a); sc[0]+=1
def arm_to(tgt,yaw=YAW,n=60,tol=0.008):
    qd,err=kin.ik(np.array(tgt), kin.rot_down(yaw), qq(), TOOL); QD[0]=qd
    for k in range(n):
        step()
        if np.abs(qd-qq()).max()<tol: break
    return err, np.abs(qd-qq()).max()
def base_to(bx,by,bth=0.0,n=40,tol=0.01):
    for k in range(n):
        b=base(); e=np.array([bx-b[0],by-b[1],bth-b[2]])
        if np.abs(e).max()<tol: break
        step(np.clip(e,-0.1,0.1))
base_to(-0.16, WALLY, 0.0)
b=base(); fx=WX-b[0]; fy=WALLY-b[1]
print('base',np.round(b,3),'fx',round(fx,3),'yellow z',round(yb()[2],4))
z0 = yb()[2]
for z in [0.26,0.24,0.22,0.20,0.18]:
    G[0]=0.0
    e,qe = arm_to((fx,fy,0.34)); 
    e,qe = arm_to((fx,fy,z),40)
    G[0]=1.0
    for i in range(10): step()
    e2,qe2 = arm_to((fx,fy,z+0.10),40)
    dz = yb()[2]-z0
    print('z',z,'ikerr',round(e,3),'qerr_desc',round(qe,3),'bin dz',round(dz,4),'bin',np.round(yb()[:3],3),'quat',np.round(yb()[3:],2),'steps',sc[0])
    if dz>0.03:
        print('LIFTED at z',z); break
    G[0]=0.0
    for i in range(5): step()
env.close()

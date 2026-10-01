from env_client import make_env
import numpy as np, kin, sys
seed=int(sys.argv[1]); oc=int(sys.argv[2]); SPD=float(sys.argv[3]); ZP=float(sys.argv[4])
env = make_env()
obs, info = env.reset(seed=seed, options={'object_count':oc})
R=obs.get_object_from_name('robot'); GB=obs.get_object_from_name('bin_green_0'); YB=obs.get_object_from_name('bin_yellow_0')
CUBES=[o for o in obs.data if o.name.startswith('cube_')]
def base(): return obs.data[R][:3].copy()
def qq(): return obs.data[R][3:10].copy()
def gb(): return obs.data[GB][:3].copy()
def yb(): return obs.data[YB][:7].copy()
def cpos(): return np.array([obs.data[c][:3] for c in CUBES])
TOOL=0.12; G=[1.0]; QD=[None]; sc=[0]; RW=[-1.0]; TERM=[False]
def step(bc=(0,0,0)):
    global obs
    a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(bc,-0.1,0.1); a[10]=G[0]
    if QD[0] is not None: a[3:10]=np.clip(QD[0]-qq(),-0.1,0.1)
    obs,rew,term,trunc,info=env.step(a); sc[0]+=1; RW[0]=rew; TERM[0]=term or trunc
def arm_to(tgt,n=60,tol=0.008,yaw=1.5708):
    qd,err=kin.ik(np.array(tgt), kin.rot_down(yaw), qq(), TOOL); QD[0]=qd
    for k in range(n):
        step()
        if np.abs(qd-qq()).max()<tol: break
    return err, np.abs(qd-qq()).max()
def base_to(bx,by,bth=0.0,n=40,tol=0.01,mv=0.1):
    for k in range(n):
        b=base(); e=np.array([bx-b[0],by-b[1],bth-b[2]])
        if np.abs(e).max()<tol: break
        step(np.clip(e,-mv,mv))
c0=cpos(); y0=yb()[:3].copy(); g0=gb().copy()
D = g0[:2]-y0[:2]
print('ideal shift', np.round(D,4))
# Phase A: evict green
base_to(-0.16, 0.0, 0.0)
fx=0.5-base()[0]
arm_to((fx,0.0,0.34)); arm_to((fx,0.0,ZP),n=40)
for i in range(8): step((0,0.1,0))
print('green at',np.round(gb(),3),'steps',sc[0])
arm_to((fx,0.0,0.40))
# Phase B: behind yellow
base_to(-0.16, y0[1]-0.375, 0.0)
fx=0.5-base()[0]
arm_to((fx,0.0,0.40)); arm_to((fx,0.0,ZP),n=40)
print('behind yellow, yellow',np.round(yb()[:3],3),'steps',sc[0])
tgt = np.array([g0[0], g0[1]])
for it in range(3):
    # push +y
    for i in range(80):
        y=yb()[1]
        if y > tgt[1]-0.004: break
        step((0, min(SPD, max(0.008,(tgt[1]-y)*0.25)), 0))
    print('after y push: yellow',np.round(yb()[:3],3),'q',np.round(yb()[3:],3),'steps',sc[0],'rew',RW[0],'term',TERM[0])
    if TERM[0]: break
    dx = tgt[0]-yb()[0]
    print(' dx needed', round(dx,4))
    if dx > 0.008:
        # push -x wall in +x direction using arm reach
        arm_to((fx,0.0,0.40))
        b=base(); ynow=yb()[1]
        base_to(-0.16, ynow, 0.0)
        b=base(); fxw = (yb()[0]-0.225-0.02)-b[0]
        arm_to((fxw,0.0,0.40)); arm_to((fxw,0.0,ZP),n=40)
        for i in range(60):
            dx = tgt[0]-yb()[0]
            if dx<0.004: break
            fxw += min(0.01, max(0.003,dx*0.2))
            arm_to((fxw,0.0,ZP),n=6)
        print(' after x push: yellow',np.round(yb()[:3],3),'steps',sc[0],'rew',RW[0])
        arm_to((fxw,0.0,0.40))
        base_to(-0.16, yb()[1]-0.375, 0.0)
        fx=0.5-base()[0]
        arm_to((fx,0.0,0.40)); arm_to((fx,0.0,ZP),n=40)
    else: break
for i in range(15): step()
c1=cpos(); sh=(c1[:,:2]-c0[:,:2])
err=np.linalg.norm(sh-D,axis=1)
print('FINAL rew',RW[0],'term',TERM[0],'steps',sc[0])
print('yellow',np.round(yb()[:3],3),'quat',np.round(yb()[3:],3))
print('cube shift err: <5cm',(err<0.05).sum(),'<3',(err<0.03).sum(),'of',len(err),'max',round(err.max(),3),'mean',round(err.mean(),3))
print('cube z',np.round(c1[:,2],3)[:6])
env.close()

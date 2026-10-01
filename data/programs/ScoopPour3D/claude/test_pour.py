from env_client import make_env
import numpy as np, kin, sys
seed=int(sys.argv[1]); oc=int(sys.argv[2]); GAPY=float(sys.argv[3]); ZG=float(sys.argv[4]); ZLIFT=float(sys.argv[5])
env = make_env()
obs, info = env.reset(seed=seed, options={'object_count':oc})
R=obs.get_object_from_name('robot'); GB=obs.get_object_from_name('bin_green_0'); YB=obs.get_object_from_name('bin_yellow_0')
CUBES=[o for o in obs.data if o.name.startswith('cube_')]
def base(): return obs.data[R][:3].copy()
def qq(): return obs.data[R][3:10].copy()
def gb(): return obs.data[GB][:3].copy()
def yb(): return obs.data[YB][:7].copy()
def cpos(): return np.array([obs.data[c][:3] for c in CUBES])
TOOL=0.12; G=[0.0]; QD=[None]; sc=[0]; RW=[-1.0]; TERM=[False]
def step(bc=(0,0,0)):
    global obs
    a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(bc,-0.1,0.1); a[10]=G[0]
    if QD[0] is not None: a[3:10]=np.clip(QD[0]-qq(),-0.1,0.1)
    obs,rew,term,trunc,info=env.step(a); sc[0]+=1; RW[0]=rew; TERM[0]=term or trunc
    return rew,term
def arm_to(tgt,yaw=1.5708,n=60,tol=0.008,bc=(0,0,0)):
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
print('green',np.round(gb(),3),'yellow',np.round(yb()[:3],3))
base_to(-0.16,-0.36,0.0)
bx=base()[0]; fx=0.5-bx
arm_to((fx,0.0,0.34)); arm_to((fx,0.0,0.21))
print('positioned steps',sc[0],'yellow',np.round(yb()[:3],3))
tgt_y = gb()[1]-0.30-GAPY
for i in range(40):
    y=yb()[1]
    if y> tgt_y-0.003: break
    step((0,min(0.06,max(0.015,(tgt_y-y)*0.4)),0))
print('pushed yellow to',np.round(yb()[:3],3),'target',round(tgt_y,3),'steps',sc[0],'rew',RW[0])
# descend to hook height and close
b=base(); fx=0.5-b[0]; fy=yb()[1]-0.155-b[1]
arm_to((fx,fy,ZG),n=40)
G[0]=1.0
for i in range(8): step()
print('hooked, yellow',np.round(yb()[:3],3),np.round(yb()[3:],2))
# lift/tip
for z in np.arange(ZG+0.03, ZLIFT, 0.03):
    arm_to((fx,fy,z),n=25)
    print(' tip z',round(z,3),'yellow',np.round(yb()[:3],3),'q',np.round(yb()[3:],2),'rew',round(RW[0],3),'term',TERM[0])
    if TERM[0]: break
for i in range(20): step()
c=cpos(); g=gb()
d=np.linalg.norm(c-g,axis=1)
print('FINAL rew',RW[0],'term',TERM[0],'steps',sc[0])
print('green',np.round(g,3),'cube mean',np.round(c.mean(0),3))
print('dist to green center 3d: <5',(d<0.05).sum(),'<10',(d<0.10).sum(),'of',len(d), np.round(np.sort(d)[:6],3))
env.close()

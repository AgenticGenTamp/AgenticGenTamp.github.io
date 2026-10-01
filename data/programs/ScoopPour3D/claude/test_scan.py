from env_client import make_env
import numpy as np, kin, sys
seed=int(sys.argv[1]); oc=int(sys.argv[2]); WX=float(sys.argv[3]); ZH=float(sys.argv[4])
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
def arm_to(tgt,Rt=None,n=60,tol=0.008,bc=(0,0,0)):
    if Rt is None: Rt = kin.rot_down(1.5708)
    qd,err=kin.ik(np.array(tgt), Rt, qq(), TOOL); QD[0]=qd
    for k in range(n):
        step(bc)
        if np.abs(qd-qq()).max()<tol: break
    return err, np.abs(qd-qq()).max()
def base_to(bx,by,bth=0.0,n=40,tol=0.01):
    for k in range(n):
        b=base(); e=np.array([bx-b[0],by-b[1],bth-b[2]])
        if np.abs(e).max()<tol: break
        step(np.clip(e,-0.1,0.1))
base_to(-0.16, -0.36, 0.0)
bx=base()[0]; fx=WX-bx
print('fx',round(fx,3))
arm_to((fx,0.0,0.34)); arm_to((fx,0.0,0.20),n=40)
G[0]=1.0
for i in range(8): step()
e,qe=arm_to((fx,0.0,ZH),n=70)
print('lift ikerr',round(e,3),'qerr',round(qe,3),'yellow',np.round(yb()[:3],3),'cubez',round(cpos()[:,2].mean(),3),'steps',sc[0])
# carry +y until cube-cluster xy is over green center
for i in range(40):
    c=cpos(); dy = gb()[1]-c[:,1].mean()
    if abs(dy)<0.01: break
    step((0,np.clip(dy*0.4,-0.06,0.06),0))
    if i%4==0:
        print(' carry',i,'cube',np.round(cpos().mean(0),3),'green',np.round(gb(),3),'rew',round(RW[0],4))
c=cpos()
print('carried cube mean',np.round(c.mean(0),3),'green',np.round(gb(),3),'rew',RW[0],'steps',sc[0])
# lower slowly
for z in np.arange(ZH-0.02, 0.30, -0.02):
    e,qe=arm_to((fx,0.0,z),n=25)
    c=cpos(); g=gb()
    d=np.linalg.norm(c-g,axis=1)
    print(' z',round(z,3),'qerr',round(qe,3),'cube',np.round(c.mean(0),3),'green',np.round(g,3),'mind',round(d.min(),3),'rew',round(RW[0],4),'term',TERM[0])
    if TERM[0]: break
env.close()

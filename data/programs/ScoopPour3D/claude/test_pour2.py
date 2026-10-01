from env_client import make_env
import numpy as np, kin, sys
seed=int(sys.argv[1]); oc=int(sys.argv[2]); ZH=float(sys.argv[3]); ROLL=float(sys.argv[4])
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
def Rx(t):
    c,s=np.cos(t),np.sin(t); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def arm_to(tgt,Rt=None,n=60,tol=0.008,bc=(0,0,0)):
    if Rt is None: Rt = kin.rot_down(1.5708)
    qd,err=kin.ik(np.array(tgt), Rt, qq(), TOOL); QD[0]=qd
    for k in range(n):
        step(bc)
        if np.abs(qd-qq()).max()<tol: break
    return err
def base_to(bx,by,bth=0.0,n=40,tol=0.01):
    for k in range(n):
        b=base(); e=np.array([bx-b[0],by-b[1],bth-b[2]])
        if np.abs(e).max()<tol: break
        step(np.clip(e,-0.1,0.1))
base_to(-0.16, -0.36, 0.0)
bx=base()[0]; fx=0.5-bx
arm_to((fx,0.0,0.34)); arm_to((fx,0.0,0.20),n=40)
G[0]=1.0
for i in range(8): step()
arm_to((fx,0.0,ZH),n=60)
print('lifted yellow',np.round(yb()[:3],3),'q',np.round(yb()[3:],2),'steps',sc[0])
# carry: move base +y so bin center over green center
for i in range(30):
    dy = gb()[1]-yb()[1]
    if abs(dy)<0.01: break
    step((0,np.clip(dy*0.5,-0.08,0.08),0))
print('carried yellow',np.round(yb()[:3],3),'green',np.round(gb(),3),'steps',sc[0],'rew',RW[0])
# pour: roll about world x
for ang in np.arange(0.2, ROLL+0.01, 0.2):
    Rt = Rx(ang) @ kin.rot_down(1.5708)
    e=arm_to((fx,0.0,ZH),Rt,n=30)
    c=cpos()
    print(' roll',round(ang,2),'ikerr',round(e,3),'yellow',np.round(yb()[:3],3),'q',np.round(yb()[3:],2),'cubez',round(c[:,2].mean(),3),'rew',round(RW[0],3),'term',TERM[0])
    if TERM[0]: break
for i in range(25): step()
c=cpos(); g=gb(); d=np.linalg.norm(c-g,axis=1)
print('FINAL rew',RW[0],'term',TERM[0],'steps',sc[0],'green',np.round(g,3))
print('cube mean',np.round(c.mean(0),3),'std',np.round(c.std(0),3))
print('d3d <5',(d<0.05).sum(),'<10',(d<0.10).sum(),'of',len(d),np.round(np.sort(d)[:8],3))
env.close()

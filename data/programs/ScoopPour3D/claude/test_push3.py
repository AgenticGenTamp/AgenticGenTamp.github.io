from env_client import make_env
import numpy as np, kin, sys
seed=int(sys.argv[1]); oc=int(sys.argv[2]); PUSH=float(sys.argv[3])
env = make_env()
obs, info = env.reset(seed=seed, options={'object_count':oc})
R = obs.get_object_from_name('robot'); GB=obs.get_object_from_name('bin_green_0'); YB=obs.get_object_from_name('bin_yellow_0')
CUBES=[o for o in obs.data if o.name.startswith('cube_')]
def base(): return obs.data[R][:3].copy()
def qq(): return obs.data[R][3:10].copy()
def gb(): return obs.data[GB][:3].copy()
def yb(): return obs.data[YB][:3].copy()
def cpos(): return np.array([obs.data[c][:3] for c in CUBES])
TOOL=0.12; GRIP=1.0
sc=[0]
QD=[None]
def step(base_cmd=(0,0,0)):
    global obs
    a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(base_cmd,-0.1,0.1); a[10]=GRIP
    if QD[0] is not None: a[3:10]=np.clip(QD[0]-qq(),-0.1,0.1)
    obs,rew,term,trunc,info = env.step(a); sc[0]+=1
    return rew,term
def arm_to(tgt, n=80, tol=0.01, base_cmd=(0,0,0)):
    qd,err = kin.ik(np.array(tgt), kin.rot_down(), qq(), TOOL); QD[0]=qd
    for k in range(n):
        step(base_cmd)
        if np.abs(qd-qq()).max()<tol: break
    return err
def base_to(bx,by,bth=0.0,n=40,tol=0.01,mv=0.1):
    for k in range(n):
        b=base(); e=np.array([bx-b[0],by-b[1],bth-b[2]])
        if np.abs(e).max()<tol: break
        step(np.clip(e,-mv,mv))
c0=cpos()
base_to(0.5,0.0,0.0)
fx=0.5-base()[0]
arm_to((fx,0.0,0.34)); arm_to((fx,0.0,0.21))
for i in range(7): step((0,0.1,0))
print('green evicted',np.round(gb(),3),'steps',sc[0])
arm_to((fx,0.0,0.40))
base_to(0.5,-0.46,0.0)
fx2=0.5-base()[0]
arm_to((fx2,0.0,0.40)); arm_to((fx2,0.0,0.21))
print('behind yellow',np.round(yb(),3),'base',np.round(base(),3),'steps',sc[0])
term=False
for i in range(80):
    y=yb()[1]
    if y > 0.198: break
    rew,term=step((0,min(PUSH,max(0.015,(0.201-y)*0.4)),0))
    if i%5==0: print(' push',i,'base_y',round(base()[1],3),'yellow',np.round(yb(),3),'rew',round(rew,4))
    if term: break
print('final yellow',np.round(yb(),3),'steps',sc[0],'term',term)
for i in range(15): rew,term=step()
c1=cpos()
print('rew',rew,'term',term,'cube mean',np.round(c1.mean(0),3),'spread',np.round(c1[:,:2].std(0),3))
print('shift',np.round((c1-c0)[:,:2].mean(0),3),'maxdev',round(np.abs((c1-c0)[:,:2]-(c1-c0)[:,:2].mean(0)).max(),3))
d=np.linalg.norm(c1[:,:2]-np.array([0.501,0.201]),axis=1)
print('dist to green center: <5cm',(d<0.05).sum(),'<10cm',(d<0.10).sum(),'of',len(d))
env.close()

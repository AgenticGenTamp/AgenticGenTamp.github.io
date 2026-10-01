from env_client import make_env
import numpy as np, kin, sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 3
oc=int(sys.argv[2]) if len(sys.argv)>2 else 10
PUSH=float(sys.argv[3]) if len(sys.argv)>3 else 0.06
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
sc=[0]; last=[-1.0]
def step(a):
    global obs
    obs,rew,term,trunc,info = env.step(a); sc[0]+=1; last[0]=rew
    return rew,term
def arm_to(tgt, n=80, tol=0.01):
    q=qq(); qd,err = kin.ik(np.array(tgt), kin.rot_down(), q, TOOL)
    for k in range(n):
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-qq(),-0.1,0.1); a[10]=GRIP
        step(a)
        if np.abs(qd-qq()).max()<tol: break
    return err
def base_to(bx,by,bth=0.0,n=40,tol=0.01,mv=0.1):
    for k in range(n):
        b=base(); e=np.array([bx-b[0],by-b[1],bth-b[2]])
        if np.abs(e).max()<tol: break
        a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(e,-mv,mv); a[10]=GRIP
        step(a)
c0 = cpos()
print('start green',np.round(gb(),3),'yellow',np.round(yb(),3),'cube spread', np.round(c0[:,:2].std(0),3), np.round(c0[:,:2].mean(0),3))
base_to(0.5,0.0,0.0)
bx=base()[0]; fx=0.5-bx
arm_to((fx,0.0,0.34)); arm_to((fx,0.0,0.21))
for i in range(7):
    a=np.zeros(11,dtype=np.float32); a[1]=0.1; a[10]=GRIP; step(a)
print('green evicted to',np.round(gb(),3),'steps',sc[0])
# retreat
arm_to((fx,0.0,0.40))
base_to(0.5,-0.46,0.0)
bx=base()[0]; fx2=0.5-bx
print('base',np.round(base(),3),'fx2',round(fx2,3))
arm_to((fx2,0.0,0.40)); arm_to((fx2,0.0,0.21))
print('behind yellow, yellow',np.round(yb(),3),'steps',sc[0])
for i in range(60):
    y=yb()[1]
    if y > 0.195: break
    a=np.zeros(11,dtype=np.float32); a[1]=min(PUSH, max(0.02,(0.201-y)*0.5)); a[10]=GRIP
    rew,term=step(a)
    if i%3==0 or term: print(' push',i,'base_y',round(base()[1],3),'yellow',np.round(yb(),3),'rew',round(rew,4),'term',term)
    if term: break
c1=cpos()
print('final yellow',np.round(yb(),3),'rew',last[0],'steps',sc[0])
print('cube mean',np.round(c1[:,:3].mean(0),3),'spread',np.round(c1[:,:2].std(0),3))
print('cube shift vs start', np.round((c1-c0)[:,:2].mean(0),3), 'max dev', np.round(np.abs((c1-c0)[:,:2]-(c1-c0)[:,:2].mean(0)).max(),3))
# settle
for i in range(10):
    rew,term=step(np.zeros(11,dtype=np.float32)); 
print('after settle rew',rew,'term',term)
env.close()

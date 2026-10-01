from env_client import make_env
import numpy as np, kin
env = make_env()
obs, info = env.reset(seed=3, options={'object_count':10})
R = obs.get_object_from_name('robot'); GB=obs.get_object_from_name('bin_green_0'); YB=obs.get_object_from_name('bin_yellow_0')
def base(): return obs.data[R][:3].copy()
def qq(): return obs.data[R][3:10].copy()
def gb(): return obs.data[GB][:3].copy()
def yb(): return obs.data[YB][:3].copy()
TOOL=0.12; GRIP=1.0
step_count=0
def step(a):
    global obs, step_count
    obs,rew,term,trunc,info = env.step(a); step_count+=1
    return rew,term
def arm_to(tgt, n=80, tol=0.01, grip=GRIP):
    q=qq()
    qd,err = kin.ik(np.array(tgt), kin.rot_down(), q, TOOL)
    for k in range(n):
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-qq(),-0.1,0.1); a[10]=grip
        step(a)
        if np.abs(qd-qq()).max()<tol: break
    return err, np.abs(qd-qq()).max()
def base_to(bx,by,bth=0.0,n=30,tol=0.01,grip=GRIP):
    for k in range(n):
        b=base(); e=np.array([bx-b[0],by-b[1],bth-b[2]])
        if np.abs(e).max()<tol: break
        a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(e,-0.1,0.1); a[10]=grip
        step(a)
    return base()
print('start base',np.round(base(),3),'green',np.round(gb(),3),'yellow',np.round(yb(),3))
print('base_to', np.round(base_to(0.5,0.0,0.0),3), 'steps', step_count)
bx,by,_=base()
fx = 0.5-bx
print('need fx', round(fx,3))
e,qe = arm_to((fx,0.0,0.34)); print('above gap ikerr',round(e,3),'qerr',round(qe,3),'steps',step_count)
e,qe = arm_to((fx,0.0,0.21)); print('in gap ikerr',round(e,3),'qerr',round(qe,3),'steps',step_count)
print('green',np.round(gb(),3),'yellow',np.round(yb(),3))
# push green +y using base
for i in range(8):
    b=base(); a=np.zeros(11,dtype=np.float32); a[1]=0.1; a[10]=GRIP
    rew,term=step(a)
    print(' push', i, 'base',np.round(base(),3),'green',np.round(gb(),3),'rew',rew)
print('steps',step_count)
env.close()

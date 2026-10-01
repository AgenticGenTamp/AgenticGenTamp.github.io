import numpy as np, threading, json
from env_client import make_env
R={}
def run(name,fn):
    try:
        env=make_env(); o,_=env.reset(seed=0); R[name]=fn(env,np.asarray(o,float)); env.close()
    except Exception as e: R[name]='ERR '+repr(e)
def A(dx=0,dy=0,dyaw=0,j=None,g=0.0):
    a=np.zeros(11); a[0]=dx;a[1]=dy;a[2]=dyaw
    if j:
        for k,v in j.items(): a[3+k]=v
    a[10]=g; return a
def roll(env,a,n):
    S=[]
    for _ in range(n):
        o,r,t,tr,_=env.step(a); S.append(np.asarray(o,float)[125:147])
    return np.array(S)
def f_diag(env,o0):
    S=roll(env,A(dx=0.1,dy=0.1),5)
    inc=np.diff(np.vstack([o0[125:128],S[:,:3]]),axis=0)
    S2=roll(env,A(dx=0.1,dy=0.1,dyaw=0.1),5)
    inc2=np.diff(np.vstack([S[-1,:3],S2[:,:3]]),axis=0)
    return {'diag_inc':np.round(inc,5).tolist(),'diag_plus_yaw_inc':np.round(inc2,5).tolist()}
def f_rotframe(env,o0):
    S=roll(env,A(dyaw=0.1),16); roll(env,A(),4)
    S0=roll(env,A(),1); yaw=float(S0[-1,2]); p0=S0[-1,:2].copy()
    S1=roll(env,A(dy=0.1),6)
    return {'yaw':round(yaw,4),'y_inc':np.round(np.diff(np.concatenate([[p0[1]],S1[:,1]])),5).tolist(),
            'x_inc':np.round(np.diff(np.concatenate([[p0[0]],S1[:,0]])),5).tolist(),
            'yaw_drift':round(float(S1[-1,2]-yaw),5)}
def f_basestop(env,o0):
    S=roll(env,A(dy=0.1),6); p=S[-1,:3].copy()
    S2=roll(env,A(),6)
    return {'after_stop_dxy_dyaw':np.round(S2[:,:3]-p,5).tolist()}
def f_rew(env,o0):
    # random exploration, look for reward variation / termination
    rng=np.random.default_rng(0); vals=set(); term=False
    for i in range(300):
        a=np.concatenate([rng.uniform(-0.1,0.1,10),[float(rng.integers(0,2))]])
        o,r,t,tr,_=env.step(a); vals.add(round(float(r),4)); term=term or t
        if t or tr: break
    return {'rewards':sorted(vals),'terminated':term,'steps':i+1}
def f_gripforce(env,o0):
    # gripper 135 exact tracking of arbitrary values
    seq=[0.0,1.0,0.3,0.75,0.2]; out=[]
    for v in seq:
        o,_,_,_,_=env.step(A(g=v)); out.append(round(float(np.asarray(o,float)[135]),5))
    return {'cmds':seq,'obs135':out}
jobs=[('diag',f_diag),('rotframe',f_rotframe),('basestop',f_basestop),('rew',f_rew),('gripf',f_gripforce)]
th=[]
for n,f in jobs:
    t=threading.Thread(target=run,args=(n,f)); t.start(); th.append(t)
for t in th: t.join()
print(json.dumps(R,indent=1))

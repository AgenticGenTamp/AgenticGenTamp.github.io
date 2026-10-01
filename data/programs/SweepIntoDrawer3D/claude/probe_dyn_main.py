import numpy as np, threading, json
from env_client import make_env

R = {}
def run(name, fn):
    try:
        env = make_env()
        o,_ = env.reset(seed=0)
        R[name] = fn(env, np.asarray(o,float))
        env.close()
    except Exception as e:
        R[name] = "ERR "+repr(e)

def A(dx=0,dy=0,dyaw=0,j=None,g=0.0):
    a=np.zeros(11); a[0]=dx; a[1]=dy; a[2]=dyaw
    if j is not None:
        for k,v in j.items(): a[3+k]=v
    a[10]=g
    return a

def roll(env,a,n):
    S=[]; RW=[]
    for _ in range(n):
        o,r,t,tr,_=env.step(a); S.append(np.asarray(o,float)[125:147]); RW.append(r)
    return np.array(S), np.array(RW)

# 1. body vs world frame
def f_frame(env,o0):
    out={}
    # rotate yaw by +0.1 for K steps
    S,_=roll(env,A(dyaw=0.1),30)
    yaw0=o0[127]; yaw1=S[-1][2]
    # stop
    S2,_=roll(env,A(),5)
    yaw2=S2[-1][2]
    p0=S2[-1][:3].copy()
    S3,_=roll(env,A(dx=0.1),20)
    p1=S3[-1][:3]
    out['yaw_start']=float(yaw0); out['yaw_after_30_dyaw']=float(yaw1); out['yaw_after_stop']=float(yaw2)
    out['xy_before_dx']=p0[:2].tolist(); out['xy_after_20_dx']=p1[:2].tolist()
    out['dxy']= (p1[:2]-p0[:2]).tolist()
    # also dy test
    S4,_=roll(env,A(),3); p2=S4[-1][:3].copy()
    S5,_=roll(env,A(dy=0.1),20); p3=S5[-1][:3]
    out['dxy_from_dy']=(p3[:2]-p2[:2]).tolist()
    out['yaw_end']=float(p3[2])
    return out

# 2. constant max command profiles
def mk_const(a,n=40):
    def f(env,o0):
        S,RW=roll(env,a,n)
        base=np.vstack([o0[125:147],S])
        return {'first10_inc':np.round(np.diff(base[:11],axis=0),5).tolist(),
                'last10_inc':np.round(np.diff(base[-11:],axis=0),5).tolist(),
                'final':np.round(S[-1],5).tolist(),
                'rew_set':sorted(set(np.round(RW,4).tolist()))}
    return f

# 3. stop test
def f_stop(env,o0):
    S,_=roll(env,A(dx=0.1,j={1:0.1}),20)
    p=S[-1].copy()
    S2,_=roll(env,A(),15)
    return {'at_stop_cmd':np.round(p,5).tolist(),
            'after_zero_steps':np.round(S2[:,[0,1,2,4,4+11]],5).tolist(),
            'settle_delta':np.round(S2[-1]-p,5).tolist()}

# 5. gripper
def f_grip(env,o0):
    S1,_=roll(env,A(g=1.0),25)
    S2,_=roll(env,A(g=0.0),25)
    S3,_=roll(env,A(g=0.5),15)
    return {'g_idx135_open_seq':np.round(S1[:,10],5).tolist(),
            'g_close_seq':np.round(S2[:,10],5).tolist(),
            'g_half_seq':np.round(S3[:,10],5).tolist()}

th=[]
jobs = [('frame',f_frame), ('basex',mk_const(A(dx=0.1))), ('yaw',mk_const(A(dyaw=0.1))),
        ('stop',f_stop), ('grip',f_grip)]
for k in range(7):
    jobs.append(('joint%d'%(k+1), mk_const(A(j={k:0.1}))))
for name,fn in jobs:
    t=threading.Thread(target=run,args=(name,fn)); t.start(); th.append(t)
for t in th: t.join()
json.dump({k:(v if isinstance(v,str) else v) for k,v in R.items()}, open('res_main.json','w'), indent=1)
print("done", list(R.keys()))

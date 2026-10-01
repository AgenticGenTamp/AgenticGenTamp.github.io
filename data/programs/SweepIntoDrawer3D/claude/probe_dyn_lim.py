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

def mk_lim(k,s):
    def f(env,o0):
        S=roll(env,A(j={k:s*0.1}),100)
        q=S[:,3+k]
        # find where it stops changing
        return {'q0':float(o0[128+k]),'qfinal':float(q[-1]),'q_last5':np.round(q[-5:],5).tolist(),
                'first_stall_idx':int(np.argmax(np.abs(np.diff(q))<1e-4)) if np.any(np.abs(np.diff(q))<1e-4) else -1}
    return f

def mk_track(k,target_delta):
    def f(env,o0):
        tgt=o0[128+k]+target_delta
        hist=[]
        n25=n10=n1=None
        for i in range(120):
            o,_,_,_,_=env.step(A(j={k:float(np.clip(tgt-cur(o0,hist,k),-0.1,0.1))}))
            o=np.asarray(o,float); hist.append(o[128+k])
            e=abs(tgt-hist[-1])
            if e<0.05 and n25 is None: n25=i+1
            if e<0.01 and n10 is None: n10=i+1
            if e<0.001 and n1 is None: n1=i+1
        return {'target':float(tgt),'final':float(hist[-1]),'n_err<0.05':n25,'n_err<0.01':n10,'n_err<0.001':n1,
                'err_traj_every5':np.round([tgt-h for h in hist[::5]],5).tolist()}
    return f
def cur(o0,hist,k): return hist[-1] if hist else o0[128+k]

def mk_base(cmd):
    def f(env,o0):
        S=roll(env,A(dx=-cmd),15)
        inc=np.diff(np.vstack([o0[125:147],S])[:,0])
        return {'cmd':-cmd,'x_inc':np.round(inc,5).tolist(),'ratio_ss':float(np.round(inc[-1]/-cmd,4))}
    return f
def f_yawcmd(env,o0):
    out={}
    for c in [0.02,0.05,0.1]:
        pass
    S=roll(env,A(dyaw=0.02),12); out['yaw_inc_c0.02']=np.round(np.diff(np.vstack([o0[125:147],S])[:,2]),5).tolist()
    return out
def f_frame2(env,o0):
    # rotate to yaw ~ pi/2 (from 3.096 rotate -0.1*? ) use dyaw negative
    S=roll(env,A(dyaw=-0.1),16); roll(env,A(),5)
    S2=roll(env,A(),1); yaw=S2[-1][2]; p0=S2[-1][:2].copy()
    S3=roll(env,A(dx=0.1),10); p1=S3[-1][:2]
    d1=(p1-p0)
    S4=roll(env,A(),3); p2=S4[-1][:2].copy()
    S5=roll(env,A(dy=0.1),10); p2b=S5[-1][:2]
    return {'yaw':float(yaw),'dxy_from_dx':np.round(d1,4).tolist(),'dxy_from_dy':np.round(p2b-p2,4).tolist(),
            'yaw_after':float(S5[-1][2])}
def f_grip2(env,o0):
    # does gripper command affect other obs? open then close, check full obs diff
    o,_,_,_,_=env.step(A(g=1.0)); a=np.asarray(o,float)
    o,_,_,_,_=env.step(A(g=0.0)); b=np.asarray(o,float)
    d=np.abs(a-b); idx=np.where(d>1e-6)[0]
    return {'changed_idx':idx.tolist(),'vals_open':np.round(a[idx],4).tolist(),'vals_closed':np.round(b[idx],4).tolist()}

jobs=[('frame2',f_frame2),('base0.02',mk_base(0.02)),('base0.05',mk_base(0.05)),('base0.1',mk_base(0.1)),
      ('yawsmall',f_yawcmd),('grip2',f_grip2),('track2',mk_track(1,1.0)),('track4',mk_track(3,-1.0)),('track0',mk_track(0,1.0))]
for k in range(7):
    jobs.append(('limP%d'%(k+1),mk_lim(k,1))); jobs.append(('limN%d'%(k+1),mk_lim(k,-1)))
th=[]
for n,f in jobs:
    t=threading.Thread(target=run,args=(n,f)); t.start(); th.append(t)
for t in th: t.join()
json.dump(R,open('res_lim.json','w'),indent=1); print('done',len(R))

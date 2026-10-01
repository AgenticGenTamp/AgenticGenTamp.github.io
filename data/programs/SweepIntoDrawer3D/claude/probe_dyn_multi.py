import numpy as np, threading, json
from env_client import make_env
R={}
def run(name,fn):
    try:
        env=make_env(); o,_=env.reset(seed=0); R[name]=fn(env,np.asarray(o,float)); env.close()
    except Exception as e: R[name]='ERR '+repr(e)
def f_drift(env,o0):
    # pure base motion in free space (+y), zero joint cmds -> do arm joints drift?
    q0=o0[128:135].copy(); mx=np.zeros(7)
    for i in range(20):
        a=np.zeros(11); a[1]=0.1; a[10]=0.0
        o,_,_,_,_=env.step(a); q=np.asarray(o,float)[128:135]
        mx=np.maximum(mx,np.abs(q-q0))
    for i in range(5):
        o,_,_,_,_=env.step(np.array([0.]*10+[0.]))
    qf=np.asarray(o,float)[128:135]
    return {'max_abs_drift_during_base_move':np.round(mx,5).tolist(),'drift_after_settle':np.round(qf-q0,5).tolist()}
def f_multi(env,o0):
    # simultaneously track 7 joint targets + base target with gain 1
    q0=o0[128:135].copy()
    tgt=q0+np.array([0.5,0.3,-0.4,0.6,-0.5,0.35,0.8])
    bt=np.array([o0[125],o0[126]+0.5,o0[127]+0.6])
    q=q0.copy(); b=o0[125:128].copy(); ns=None
    for i in range(80):
        a=np.zeros(11)
        eb=bt-b; a[0]=np.clip(eb[0]/0.87,-0.1,0.1); a[1]=np.clip(eb[1]/0.87,-0.1,0.1); a[2]=np.clip(eb[2]/0.994,-0.1,0.1)
        a[3:10]=np.clip(tgt-q,-0.1,0.1); a[10]=1.0
        o,_,_,_,_=env.step(a); o=np.asarray(o,float); q=o[128:135]; b=o[125:128]
        if ns is None and np.max(np.abs(tgt-q))<0.005 and np.max(np.abs(bt-b))<0.005: ns=i+1
    return {'steps_to_converge_all':ns,'joint_err':np.round(tgt-q,5).tolist(),'base_err':np.round(bt-b,5).tolist(),
            'grip':round(float(o[135]),3)}
th=[]
for n,f in [('drift',f_drift),('multi',f_multi)]:
    t=threading.Thread(target=run,args=(n,f)); t.start(); th.append(t)
for t in th: t.join()
print(json.dumps(R,indent=1))

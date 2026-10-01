import numpy as np, threading, json
from env_client import make_env
R={}
def run(name,fn):
    try:
        env=make_env(); o,_=env.reset(seed=0); R[name]=fn(env,np.asarray(o,float)); env.close()
    except Exception as e: R[name]='ERR '+repr(e)
def act(k,u):
    a=np.zeros(11); a[3+k]=u; return a
def mk(k,delta,gain,n=80):
    def f(env,o0):
        tgt=o0[128+k]+delta; q=o0[128+k]; traj=[]
        n5=n1=None
        for i in range(n):
            u=float(np.clip(gain*(tgt-q),-0.1,0.1))
            o,_,_,_,_=env.step(act(k,u)); q=float(np.asarray(o,float)[128+k]); traj.append(q)
            e=abs(tgt-q)
            if e<0.01 and n5 is None: n5=i+1
            if e<0.002 and n1 is None: n1=i+1
        # settle: 5 zero steps
        for _ in range(5):
            o,_,_,_,_=env.step(act(k,0.0)); q=float(np.asarray(o,float)[128+k])
        return {'gain':gain,'delta':delta,'n_err<0.01':n5,'n_err<0.002':n1,'final_err':round(tgt-traj[-1],5),
                'err_after_settle':round(tgt-q,5),'err_every4':np.round([tgt-t for t in traj[:32:4]],4).tolist()}
    return f
jobs=[]
for g in [1,2,3,4,5]:
    jobs.append(('big_g%d'%g,mk(0,1.0,g)))
    jobs.append(('small_g%d'%g,mk(0,0.05,g,40)))
th=[]
for n,f in jobs:
    t=threading.Thread(target=run,args=(n,f)); t.start(); th.append(t)
for t in th: t.join()
for k in sorted(R): print(k,R[k])

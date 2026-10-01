import numpy as np, threading, json
from env_client import make_env
R={}
def run(name,fn):
    try:
        env=make_env(); o,_=env.reset(seed=0); R[name]=fn(env,np.asarray(o,float)); env.close()
    except Exception as e: R[name]='ERR '+repr(e)
def mk(g):
    def f(env,o0):
        bt=np.array([o0[125],o0[126]+0.4,o0[127]+0.5]); b=o0[125:128].copy(); ns=None; hist=[]
        for i in range(40):
            a=np.zeros(11); e=bt-b
            a[0]=np.clip(g*e[0],-0.1,0.1); a[1]=np.clip(g*e[1],-0.1,0.1); a[2]=np.clip(e[2]/0.994,-0.1,0.1)
            o,_,_,_,_=env.step(a); b=np.asarray(o,float)[125:128]
            hist.append(np.round(bt-b,4).tolist())
            if ns is None and np.max(np.abs(bt-b))<0.002: ns=i+1
        return {'gain':g,'n_conv':ns,'final_err':np.round(bt-b,5).tolist(),'err_first6':hist[:6]}
    return f
th=[]
for g in [1.0,1/0.87]:
    t=threading.Thread(target=run,args=('g%.3f'%g,mk(g))); t.start(); th.append(t)
for t in th: t.join()
print(json.dumps(R,indent=1))

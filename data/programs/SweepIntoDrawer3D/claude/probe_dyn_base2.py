import numpy as np, json
from env_client import make_env
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
bt=np.array([o[125],o[126]+0.4]); b=o[125:127].copy(); hist=[]; ns=None
for i in range(20):
    a=np.zeros(11); e=bt-b
    a[0]=np.clip(e[0]/0.8704,-0.1,0.1); a[1]=np.clip(e[1]/0.8704,-0.1,0.1)
    o,_,_,_,_=env.step(a); b=np.asarray(o,float)[125:127]; hist.append(round(float(bt[1]-b[1]),5))
    if ns is None and abs(bt[1]-b[1])<0.001: ns=i+1
print('pure translation gain 1/0.8704: n_conv',ns,'errs',hist[:10],'final',hist[-1])
env.close()

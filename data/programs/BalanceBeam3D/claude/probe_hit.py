import numpy as np
from env_client import make_env
env=make_env(); rng=np.random.default_rng(1)
obs,_=env.reset(seed=0); o=np.asarray(obs,float)
ssx,ssy=o[38],o[39]
a=np.zeros(11,dtype=np.float32)
# drive base toward seesaw
for t in range(200):
    o=np.asarray(obs,float)
    dx=np.clip((ssx-0.45)-o[16],-0.1,0.1); dy=np.clip(ssy-o[17],-0.1,0.1)
    a[:]=0; a[0]=dx; a[1]=dy
    if abs(dx)<1e-3 and abs(dy)<1e-3: break
    obs,r,te,tr,_=env.step(a)
print("base",np.round(np.asarray(obs,float)[16:19],3),"r",r)
# now flail arm with big deltas
seen=set()
for t in range(300):
    a[:]=0; a[3:10]=rng.uniform(-0.1,0.1,7); a[10]=1.0
    obs,r,te,tr,_=env.step(a); o=np.asarray(obs,float)
    seen.add(round(r,4))
    if te or tr: print("END",t,te,tr,r); break
print("rewards seen",sorted(seen))
print("ss pos/quat",np.round(o[38:45],4),"ssvel",np.round(o[45:51],3))
print("blocks",np.round(o[0:3],3),np.round(o[54:57],3),np.round(o[70:73],3))
print("ee joints",np.round(o[19:27],3))

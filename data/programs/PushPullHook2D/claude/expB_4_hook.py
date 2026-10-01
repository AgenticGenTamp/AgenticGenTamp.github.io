import numpy as np
from helper import Sim
def probe(s,d,steps=(0.01,0.002,0.0004,0.0001),maxn=900):
    d=np.array(d,float)
    for st in steps:
        for i in range(maxn):
            p=s.obs[:2].copy(); s.step([d[0]*st,d[1]*st,0,0,1.0])
            if np.linalg.norm(s.obs[:2]-p)<1e-12: break
    return s.obs.copy()
def hookpts(o):
    C=o[9:11]; th=o[11]; a=np.array([-np.cos(th),-np.sin(th)]); b=np.array([np.sin(th),-np.cos(th)])
    return C, C+a*o[18], C+b*o[19]
s=Sim(42)
ok=s.grasp_hook(1.15)
o=s.obs.copy()
print("after grasp rob=",np.round(o[[0,1,2,4,6]],4)," hook=",np.round(o[9:12],4))
# check rigid attachment: small move
p0=o.copy(); s.step([0.02,0.0,0.0,0,1.0]); o1=s.obs
print("dmove rob",np.round(o1[[0,1,2]]-p0[[0,1,2]],4),"hook",np.round(o1[9:12]-p0[9:12],4))
s.step([0,0,0.1,0,1.0]); o2=s.obs
print("drot rob",np.round(o2[[0,1,2]]-o1[[0,1,2]],4),"hook",np.round(o2[9:12]-o1[9:12],4))
# offset in robot frame
def offs(o):
    th=o[2]; R=np.array([[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]])
    return R@(o[9:11]-o[:2]), (o[11]-o[2])
print("hook offset in robot frame",np.round(offs(o2)[0],4),"dth",round(offs(o2)[1],4))
np.save("expB_grasped42.npy", s.obs)
s.close()

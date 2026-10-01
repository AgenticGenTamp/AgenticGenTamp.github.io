import sys, numpy as np
from helper import Sim, wrap, R
from expA_lib import rel, robot_for
def setup(sd,gap,d=1.15,hth=np.pi/2,dy=0.30):
    s=Sim(sd); s.grasp_hook(d); o=s.obs
    Cl,dth=rel(s); M=o[20:22].copy()
    Cdes=np.array([M[0]-gap, M[1]+dy]); Rp,rth=robot_for(Cl,dth,Cdes,hth)
    s.goto(o[0],o[1],rth,0.2,1.0); s.goto(Rp[0],s.obs[1],rth,0.2,1.0); s.goto(Rp[0],Rp[1],rth,0.2,1.0)
    return s
print("== fine approach inc=0.002 ==")
s=setup(42,0.16)
for i in range(40):
    p=s.obs[[9,20]].copy(); s.step([0.002,0,0,0,1.0]); o=s.obs
    pre=p[1]-(p[0]+0.002); post=o[20]-o[9]
    if abs(o[20]-p[1])>1e-7: print(" i%2d pre_sep=%.4f jump=%.4f post=%.4f"%(i,pre,o[20]-p[1],post))
print("== stationary while in band: move to sep 0.11 then 20 zero-x steps ==")
o=s.obs; print(" sep now %.4f"%(o[20]-o[9]))
tgt=o[20]-0.11
while s.obs[9]<tgt-1e-4:
    s.step([min(0.002,tgt-s.obs[9]),0,0,0,1.0])
o=s.obs; print(" sep=%.4f"%(o[20]-o[9]))
M=s.obs[20:22].copy()
for i in range(10): s.step([0,0,0,0,1.0])
print(" after 10 null steps dM=",np.round(s.obs[20:22]-M,5)," sep=%.4f"%(s.obs[20]-s.obs[9]))
print("== retract (pull away) does button follow? ==")
M=s.obs[20:22].copy()
for i in range(10): s.step([-0.01,0,0,0,1.0])
print(" after 10 x-0.01 dM=",np.round(s.obs[20:22]-M,5)," sep=%.4f"%(s.obs[20]-s.obs[9]))
s.close()

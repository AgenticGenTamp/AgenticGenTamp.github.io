import numpy as np
from helper import Sim
from expA_lib import rel, robot_for
def setup(sd,gap,d=1.15,hth=np.pi/2,dy=0.30):
    s=Sim(sd); s.grasp_hook(d); o=s.obs
    Cl,dth=rel(s); M=o[20:22].copy()
    Cdes=np.array([M[0]-gap, M[1]+dy]); Rp,rth=robot_for(Cl,dth,Cdes,hth)
    s.goto(o[0],o[1],rth,0.2,1.0); s.goto(Rp[0],s.obs[1],rth,0.2,1.0); s.goto(Rp[0],Rp[1],rth,0.2,1.0)
    return s
s=setup(42,0.30)
def approach_to(sep):
    while s.obs[20]-s.obs[9] > sep+1e-6:
        d=min(0.005, s.obs[20]-s.obs[9]-sep); s.step([d,0,0,0,1.0])
print("h      pre_sep  jumpx    jumpy   post_sep")
for h in [0.001,0.005,0.01,0.02,0.03,0.04,0.045,0.05,0.055,0.06,0.07,0.08,0.09]:
    approach_to(0.1005)
    p=s.obs[[9,20,21]].copy(); s.step([h,0,0,0,1.0]); o=s.obs
    pre=p[1]-(p[0]+h)
    print("%.3f  %.4f  %+.4f  %+.5f  %.4f"%(h,pre,o[20]-p[1],o[21]-p[2],o[20]-o[9]))
    # reset bar behind button
    while s.obs[20]-s.obs[9] < 0.30: s.step([-0.05,0,0,0,1.0])
s.close()

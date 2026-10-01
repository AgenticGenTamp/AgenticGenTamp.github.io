import numpy as np
from helper import Sim
for sd in [42,0,1,3,7]:
    s=Sim(sd); o=s.obs
    print(f"seed{sd} R=({o[0]:.3f},{o[1]:.3f},th={o[2]:.3f}) hook=({o[9]:.3f},{o[10]:.3f},th={o[11]:.3f}) M=({o[20]:.3f},{o[21]:.3f}) T=({o[29]:.3f},{o[30]:.3f}) |MT|={np.hypot(o[20]-o[29],o[21]-o[30]):.3f}")
    s.close()

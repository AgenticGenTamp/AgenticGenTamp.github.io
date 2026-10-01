import numpy as np
from helper import Sim
for seed in [42]:
    s=Sim(seed); o=s.obs
    print("seed",seed)
    print(" robot xy th arm",np.round(o[[0,1,2,4]],4))
    print(" hook",np.round(o[9:12],4),"w",o[17],"l1",o[18],"l2",o[19])
    print(" mov",np.round(o[20:22],4),o[28],"tgt",np.round(o[29:31],4),o[37])
    print(" obs len",len(o))
    s.close()

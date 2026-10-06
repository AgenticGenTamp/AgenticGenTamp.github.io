import numpy as np, approach
from approach import Kin, down_rot, HOME, wrap
k=Kin()
for r in [0.35,0.4,0.45,0.5,0.55,0.6]:
    out=[]
    for z in [0.15,0.07,0.015]:
        q,pe,re_=k.ik(np.zeros(3), np.array([r,0,z]), down_rot(np.pi/2), HOME, bias=HOME)
        out.append((np.round(q,2), round(pe,4), round(re_,4), round(np.max(np.abs([wrap(x) for x in q-HOME])),2)))
    print(r, out[0][3], out[1][3], out[2][3], out[2][1], out[2][2]); print("   ", out[0][0], out[2][0])

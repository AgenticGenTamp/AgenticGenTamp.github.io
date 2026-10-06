import numpy as np, time
from kin import Kin, rot_pitch
k=Kin()
q0=np.array([0,-0.349,3.142,-2.548,0,-0.873,1.571])
print(k.cube_center(q0))
t=time.time()
q,ok=k.ik(np.array([0.45,0,0.02]), rot_pitch(0.3), q0)
print(ok, q.round(3), time.time()-t, k.cube_center(q)[0])

import numpy as np
from kin import Kin, rot_pitch
k=Kin()
q0=np.array([0,-0.349,3.142,-2.548,0,-0.873,1.571])
seeds=[q0, np.array([0,0.5,3.142,-2.0,0,-1.0,1.571]), np.array([0,1.0,3.142,-1.5,0,0.5,1.571])]
tha=-0.2; thb=tha+np.pi/2; ra=0.36; rb=0.88
for sd in seeds:
    qa,ok=k.ik(np.array([ra,0,0.02]),rot_pitch(tha),sd)
    if ok: break
qh,ok2=k.ik(np.array([ra,0,0.10]),rot_pitch(tha),qa,null_target=qa)
qb,ok3=k.ik(np.array([rb,0,0.578]),rot_pitch(thb),qa,null_target=qa)
print(ok,ok2,ok3); print("qa",list(qa.round(4))); print("qh",list(qh.round(4))); print("qb",list(qb.round(4)))
print("dq a-b",(qb-qa).round(3),"a-h",(qh-qa).round(3))
pts=k.arm_points(qb)
allp=np.vstack(pts)
for depth in [0.0,0.08,0.16]:
    front=rb-(1.42-1.373)-depth-0.01
    ins=allp[allp[:,0]>front]
    print(depth, "inside pts z range", ins[:,2].min().round(3), ins[:,2].max().round(3), "y range", ins[:,1].min().round(3), ins[:,1].max().round(3))
# lowest points at pick
pts=k.arm_points(qa); allp=np.vstack(pts); print("pick min z", allp[:,2].min().round(4))
print("finger y extents at pick (pts near pinch)", )

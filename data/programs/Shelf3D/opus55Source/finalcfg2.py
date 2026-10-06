import numpy as np, sys
from kin import Kin, rot_pitch
k=Kin()
q0=np.array([0,-0.349,3.142,-2.548,0,-0.873,1.571])
seeds=[q0, np.array([0,0.5,3.142,-2.0,0,-1.0,1.571]), np.array([0,1.0,3.142,-1.5,0,0.5,1.571])]
tha=float(sys.argv[1]); ra=float(sys.argv[2]); rb=float(sys.argv[3]); thb=tha+np.pi/2
for sd in seeds:
    qa,ok=k.ik(np.array([ra,0,0.022]),rot_pitch(tha),sd)
    if ok: break
qb,ok3=k.ik(np.array([rb,0,0.578]),rot_pitch(thb),qa,null_target=qa)
print(ok,ok3); print("qa",list(qa.round(4))); print("qb",list(qb.round(4))); print("dq",(qb-qa).round(3), np.abs(qb-qa).max())
allp=np.vstack(k.arm_points(qb))
for depth in [0.0,0.09,0.17]:
    front=rb-(1.43-1.373)-depth-0.01
    ins=allp[allp[:,0]>front]
    print(depth, "inside z", ins[:,2].min().round(3), ins[:,2].max().round(3))
allp=np.vstack(k.arm_points(qa)); print("pick min z", allp[:,2].min().round(4))

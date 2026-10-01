import numpy as np
from th import H
from approach import GeneratedApproach
from kin import fk
h=H(215,oc=5)
ap=GeneratedApproach(None,None,{}); ap.reset(h.obs,{})
for t in range(49):
    a=ap.get_action(h.obs); h.step(a)
print('grasped',h.grasped(), 'yaw', ap.task['yaw'])
M=h.tool(); p=M[:3,3]; print('tool',p.round(3))
print('grasp_tf', [round(h.obs.get(h.R,f),3) for f in ['grasp_tf_x','grasp_tf_y','grasp_tf_z']])
for n in ['obstruction2','obstruction1','obstruction0','obstruction4']: print(n,h.pose(n)[:3].round(3), h.he(n).round(3))
q0=h.q()
from kin import ik, down_R
y=ap.task['yaw']
for d in [(0,0,0.01),(0.01,0,0),(-0.01,0,0),(0,0.01,0),(0,-0.01,0),(0,0,-0.003)]:
    qt,e=ik(h.base(),q0,p+np.array(d),down_R(y))
    a=np.zeros(11); a[3:10]=np.clip(qt-q0,-.2,.2); h.step(a)
    moved=not np.allclose(h.q(),q0)
    print(d,'moved',moved)
    if moved:
        a=np.zeros(11); a[3:10]=np.clip(q0-h.q(),-.2,.2); h.step(a)

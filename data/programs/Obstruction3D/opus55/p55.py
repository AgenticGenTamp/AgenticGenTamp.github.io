import numpy as np
from th import H
from approach import GeneratedApproach
h=H(55)
ap=GeneratedApproach(None,None,{}); ap.reset(h.obs,{})
for t in range(4):
    a=ap.get_action(h.obs); h.step(a)
    print(t, ap.phase, h.tool()[:3,3].round(3), [ (np.round(w[0],3)) for w in ap.path])
print(ap.task, ap.grasp_off)
from kin import ik, fk
q=h.q(); print('q',q.round(2))
for z in [0.215,0.225,0.239]:
    qt,e=ap._ik(np.array([0.21,-0.003,z]),ap.task['yaw']); print(z,'err',e, 'qt',qt.round(2), 'fk',fk(ap.base,qt)[:3,3].round(3))
print('obj', h.pose('obstruction1')[:3].round(3), 'grasped', h.grasped())
for z in [0.239,0.225,0.215]:
    q=h.q(); qt,e=ap._ik(np.array([0.21,-0.003,z]),ap.task['yaw'])
    a=np.zeros(11); a[3:10]=qt-q; h.step(a); print(z,'moved', not np.allclose(h.q(),q), h.tool()[:3,3].round(3))
a=np.zeros(11); a[10]=1; h.step(a); print('open ->grasped', h.grasped(), h.pose('obstruction1')[:3].round(3))

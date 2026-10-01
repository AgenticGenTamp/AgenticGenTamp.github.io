import numpy as np
from th import H
from approach import GeneratedApproach
from kin import fk, ik, down_R, rot_err
h=H(215,oc=5)
ap=GeneratedApproach(None,None,{}); ap.reset(h.obs,{})
for t in range(49):
    a=ap.get_action(h.obs); h.step(a)
a=ap.get_action(h.obs)
print('path', [ (np.round(w[0],3) if isinstance(w,tuple) else np.round(w,3)) for w in ap.path][:6])
print('action dq', a[3:10].round(3), 'grip', a[10])
M=h.tool(); print('cur yaw', np.arctan2(M[1,0],M[0,0]), 'ori err', rot_err(M[:3,:3], down_R(ap.task['yaw'])).round(3))
q0=h.q()
for frac in [1,0.5,0.25,0.1,0.05]:
    b=a.copy(); b[3:10]*=frac; h.step(b); moved=not np.allclose(h.q(),q0)
    print(frac,'moved',moved, h.tool()[:3,3].round(3))
    if moved: break
q1=h.q(); p=h.tool()[:3,3]; y=ap.task['yaw']
print('grasp_tf', [round(h.obs.get(h.R,f),3) for f in ['grasp_tf_x','grasp_tf_y','grasp_tf_z']], 'obj', h.pose('obstruction2')[:3].round(3))
for d in [(0,0,0.005),(0,0,-0.005),(0.005,0,0),(-0.005,0,0),(0,0.005,0),(0,-0.005,0)]:
    qt,e=ik(h.base(),q1,p+np.array(d),down_R(y))
    b=np.zeros(11); b[3:10]=np.clip(qt-q1,-.2,.2); h.step(b)
    moved=not np.allclose(h.q(),q1); print(d,'moved',moved)
    if moved:
        b=np.zeros(11); b[3:10]=np.clip(q1-h.q(),-.2,.2); h.step(b); print(' back', np.allclose(h.q(),q1))

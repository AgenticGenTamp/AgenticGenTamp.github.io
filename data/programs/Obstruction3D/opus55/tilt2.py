import numpy as np
from kin import ik, down_R, fk
from th import H
def Rax(ax,a):
    ax=np.asarray(ax,float); K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
h=H(312,oc=5); base=h.base()
n='obstruction3'; p=h.pose(n)[:3]; he=h.he(n)
R=Rax((0,1,0),0.3)@down_R(np.pi/2)
def go(pos):
    for k in range(40):
        q=h.q(); qt,e=ik(base,q,np.asarray(pos,float),R,iters=100)
        d=qt-q; m=np.abs(d).max()
        if m<1e-4: break
        a=np.zeros(11); a[3:10]=d*min(1,0.2/m); h.step(a)
        if np.allclose(h.q(),q): print(' blocked'); break
    print('at',h.tool()[:3,3].round(3),'target',np.round(pos,3), 'e',e)
top=p[2]+he[2]
go((p[0],p[1],0.25)); go((p[0],p[1],top+0.03))
h.grip(-1); print('grasped',h.grasped(), [round(h.obs.get(h.R,f),3) for f in ['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']])
go((p[0]+0.15,p[1],0.25)); go((p[0]+0.15,p[1],top+0.03+0.001))
print('obj pose', h.pose(n).round(3))
h.grip(1); print('grasped after open',h.grasped(), h.pose(n).round(3))

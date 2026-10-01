import numpy as np, sys
from kin import ik, down_R, fk
from th import H
def Rax(ax,a):
    ax=np.asarray(ax,float); K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
tilt=float(sys.argv[1]); dz=float(sys.argv[2]); yaw=float(sys.argv[3])
h=H(312,oc=5); base=h.base()
n='target_block'; p=h.pose(n)[:3]; he=h.he(n)
R=Rax((0,1,0),tilt)@down_R(yaw)
def go(pos):
    for k in range(40):
        q=h.q(); qt,e=ik(base,q,np.asarray(pos,float),R,iters=100)
        d=qt-q; m=np.abs(d).max()
        if m<1e-4: break
        a=np.zeros(11); a[3:10]=d*min(1,0.2/m); h.step(a)
        if np.allclose(h.q(),q): print(' blocked'); break
    return h.tool()[:3,3].round(3)
top=p[2]+he[2]
go((p[0],p[1],0.25)); print(go((p[0],p[1],top+dz)))
h.grip(-1); print('tilt',tilt,'dz',dz,'grasped',h.grasped(), [round(h.obs.get(h.R,f),3) for f in ['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']])
if h.grasped():
    print(go((p[0]-0.1,p[1],top+dz+0.001)), h.pose(n).round(3)); h.grip(1); print('released', not h.grasped(), h.pose(n).round(3))

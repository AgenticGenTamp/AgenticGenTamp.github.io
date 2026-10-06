import numpy as np, sys, mujoco
from approach import Kin, down_R, axis_R
k=Kin(); m=k.m
bids=[i for i in range(m.nbody) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) or '').startswith('robot_') and 'base' not in mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i)]
Gp=np.array([-0.215,0,0.021]); R0=down_R(0.0)
for sgn in [-1,1]:
 for zc in [0.72,0.8,0.88]:
  for by in [0.0,0.1]:
    c=np.array([0.5,0.2,zc]); base=np.array([-0.16,by,0])
    q=np.array([0.27,0.12,2.5,-1.7,0.07,-1.35,-0.38]); out=[]
    for phi in np.radians(np.arange(0,170,15))*sgn:
        Rb=axis_R(np.array([1.,0,0]),phi)
        q,e=k.ik(base,q,c+Rb@Gp,Rb@R0,iters=200)
        k.fk(base,q); zs=k.d.xpos[bids]
        out.append((round(e,3), round(zs[:,2].min(),2), round(abs(q[1]),2),round(abs(q[3]),2),round(abs(q[5]),2)))
    print(sgn,zc,by,'maxerr',max(o[0] for o in out),'minz',min(o[1] for o in out),'j2max',max(o[2] for o in out),'j6max',max(o[4] for o in out))
print('---')
for zc in [0.72,0.85]:
  c=np.array([0.5,0.2,zc]); base=np.array([-0.16,0.1,0])
  q=np.array([0.27,0.12,2.5,-1.7,0.07,-1.35,-0.38])
  for phi in np.radians([0,-60,-90,-120,-165]):
    Rb=axis_R(np.array([1.,0,0]),phi); q,e=k.ik(base,q,c+Rb@Gp,Rb@R0,iters=200); k.fk(base,q)
    print(zc,round(np.degrees(phi)),[(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b)[6:16],np.round(k.d.xpos[b],2).tolist()) for b in bids if k.d.xpos[b][0]>0.05])

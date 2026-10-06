import numpy as np, mujoco
import approach as A
K=A.Kin(A._get_model())
b=np.array([1.4,0,np.pi])
for r in [0.55,0.6,0.65,0.7]:
  for t in [0.0,0.2,0.4,0.6]:
    R=A.R_down(t)
    q=K.ik_multi(b,A.Q_HOME,np.array([1.4-r,0,0.55]),R)
    if q is None: print(r,t,None); continue
    q2,ok=K.ik(b,q,np.array([1.4-r,0,0.495]),R)
    K.fk(b,q2); mujoco.mj_comPos(K.m,K.d); mujoco.mj_jacSite(K.m,K.d,K.jp,K.jr,K.sid)
    J=np.vstack([K.jp[:,49:56],K.jr[:,49:56]]); sv=np.linalg.svd(J,compute_uv=False)
    print(r,t,ok,q.round(2),'dq',np.abs(q2-q).max().round(3),'minsv',sv.min().round(3))

import sys; sys.path.insert(0,'/sandbox/scratch'); from h import *
from scipy.spatial.transform import Rotation as R
h=H(32); h.step(60); m,d=h.m,h.d
b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"bowl_0"); qa=m.jnt_qposadr[m.body_jntadr[b]]; da=m.jnt_dofadr[m.body_jntadr[b]]
dp=h.drink(); qq=R.from_rotvec(np.array([0,1.,0])*np.radians(30)).as_quat()
d.qpos[qa:qa+3]=[dp[0]+0.05,dp[1],0.518]; d.qpos[qa+3:qa+7]=[qq[3],*qq[:3]]; d.qvel[da:da+6]=0; mujoco.mj_forward(m,d)
for t in range(12):
    h.step(25); print(t*25, np.round(d.xpos[b],3), np.round(d.xmat[b].reshape(3,3)[:,2],2), "ncon", sum(1 for i in range(d.ncon) if b in (m.geom_bodyid[d.contact[i].geom1],m.geom_bodyid[d.contact[i].geom2])))
np.save("lean32.npy", h.obs)

from ik_tool import *
import numpy as np
thb = m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,'robot_joint_th')]
b = m.site_bodyid[site]; chain=[]
while b != thb:
    chain.append(b); b = m.body_parentid[b]
chain=chain[::-1]
rows=[]
for b in chain:
    js=[j for j in range(m.njnt) if m.jnt_bodyid[j]==b]
    jn=None; ax=[0,0,1]
    for j in js:
        n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
        if n and n.startswith('robot_joint_'): jn=int(n.split('_')[-1]); ax=m.jnt_axis[j].tolist(); assert np.allclose(m.jnt_pos[j],0)
        else: print('other joint', n)
    rows.append((m.body_pos[b].round(6).tolist(), m.body_quat[b].round(6).tolist(), jn, ax))
print('FK_CHAIN =', rows)
print('FK_SITE =', m.site_pos[site].round(6).tolist(), m.site_quat[site].round(6).tolist())

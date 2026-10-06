import numpy as np, mujoco
from run_test import env
from approach import GeneratedApproach as G, arm_q
s,info=env.reset(seed=0,options={'object_count':1})
oc=env._current_backend._object_centric_env; m=oc._robot_env.sim.model.mj_model; d=oc._robot_env.sim.data.mj_data
# find arm joint qpos addrs
names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(m.njnt)]
aj=[names.index('robot_joint_%d'%k) for k in range(1,8)]
print([names[j] for j in aj])
for z in (0.035,0.07,0.08):
    q=arm_q(z)
    for k,j in enumerate(aj): d.qpos[m.jnt_qposadr[j]]=q[k]
    mujoco.mj_forward(m,d)
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_base_link')
    bp=d.xpos[bid]; bR=d.xmat[bid].reshape(3,3)
    rows=[]
    for g in range(m.ngeom):
        b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g])
        if not b or not b.startswith('robot') or 'base' in b: continue
        if not (m.geom_contype[g]|m.geom_conaffinity[g]): continue
        # sample vertices via aabb
        c=bR.T@(d.geom_xpos[g]-bp); R=bR.T@d.geom_xmat[g].reshape(3,3)
        a=m.geom_aabb[g][3:]; ac=m.geom_aabb[g][:3]
        cc=c+R@ac; e=np.abs(R)@a
        rows.append((b,(cc-e).round(3),(cc+e).round(3)))
    print('z',z)
    for r in sorted(rows,key=lambda r:-r[2][0])[:8]: print('  ',r)

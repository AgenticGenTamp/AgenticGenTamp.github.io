import time, numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
oc = env._kinder_env._object_centric_env
env.reset(seed=0)
re = oc._robot_env; m, d = re.sim.model.mj_model, re.sim.data.mj_data
# island geom AABB
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "kitchen_island")
lo=np.full(3,1e9); hi=-lo
for g in range(m.ngeom):
    if m.geom_bodyid[g]==bid and m.geom_contype[g]|m.geom_conaffinity[g]:
        c=d.geom_xpos[g]; R=d.geom_xmat[g].reshape(3,3); s=m.geom_size[g]
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
            e=np.abs(R)@s; lo=np.minimum(lo,c-e); hi=np.maximum(hi,c+e)
        elif m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            r=m.geom_rbound[g]; lo=np.minimum(lo,c-r); hi=np.maximum(hi,c+r)
print("island collision AABB", np.round(lo,3), np.round(hi,3))
# base geoms
rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot_base_link")
for g in range(m.ngeom):
    if m.geom_bodyid[g]==rb: print("base geom type",m.geom_type[g],"size",np.round(m.geom_size[g],3),"pos",np.round(d.geom_xpos[g],3),"contype",m.geom_contype[g])
print("qpos addrs base", [m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in ["robot_joint_x","robot_joint_y","robot_joint_th","robot_joint_1","robot_joint_7"]])
for n in ["robot_joint_%d"%i for i in range(1,8)]:
    j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n); print(n,"limited",m.jnt_limited[j],np.round(m.jnt_range[j],2))
# friction of objects
for name in ["bowl_0","boxed_drink_0","can_0"]:
    b=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,name)
    gs=[g for g in range(m.ngeom) if m.geom_bodyid[g]==b and (m.geom_contype[g] or m.geom_conaffinity[g])]
    print(name,"mass",round(float(m.body_mass[b]),3),"ncollgeoms",len(gs),"friction",np.round(m.geom_friction[gs[0]],2) if gs else None)
# closed-loop tracking to target arm config
st=lambda: (np.array(re.qpos["arm"]).copy(), np.array(re.qpos["base"]).copy())
q,_=st(); tgt=q+np.array([0.5,0.4,0,0.5,0,0.3,0])
for k in range(30):
    q,b=st(); a=np.zeros(11); a[3:10]=np.clip(tgt-q,-0.1,0.1); env.step(a)
    if k%5==4: print(k+1,"err",np.round(np.abs(tgt-st()[0]).max(),4))
# absolute-ish: send full error unclipped? (action space clip is not enforced?)
q,_=st(); tgt=q+np.array([-0.5,0,0,0,0,0,0])
a=np.zeros(11); a[3:10]=tgt-q; env.step(a); print("unclipped delta 0.5 one step -> moved", np.round(st()[0]-q,3)[0])

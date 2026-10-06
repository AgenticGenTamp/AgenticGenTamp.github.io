import numpy as np, mujoco, sys
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
seed=int(sys.argv[1]) if len(sys.argv)>1 else 32
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
obs,_=env.reset(seed=seed)
re = env._kinder_env._object_centric_env._robot_env; m, d = re.sim.model.mj_model, re.sim.data.mj_data
def gn(i): return mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or f"g{i}"
def bn(i): return mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i)
bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"boxed_drink_0")
print("drink pos",d.xpos[bid],"quat",d.xquat[bid])
for g in range(m.ngeom):
    if m.geom_bodyid[g]==bid: print(" drink geom",gn(g),m.geom_type[g],m.geom_size[g],d.geom_xpos[g],"contype",m.geom_contype[g],m.geom_conaffinity[g])
def contacts():
    for i in range(d.ncon):
        c=d.contact[i]; b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
        if bid in (b1,b2): print("  con",gn(c.geom1),bn(b1),"|",gn(c.geom2),bn(b2),"pos",np.round(c.pos,3),"dist",round(c.dist,4),"n",np.round(c.frame[:3],2))
print("contacts at reset"); mujoco.mj_forward(m,d); contacts()
# joints
for j in range(m.njnt):
    n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
    if n and ('island' in n or 'drawer' in n or 'kitchen' in n): print("joint",n,"type",m.jnt_type[j],"qpos",d.qpos[m.jnt_qposadr[j]],"range",m.jnt_range[j])
# island geoms
print("island-ish geoms near drink (|x-dx|<1, |y-dy|<0.6)")
dp=d.xpos[bid].copy()
for g in range(m.ngeom):
    b=bn(m.geom_bodyid[g])
    p=d.geom_xpos[g]
    if b and ('island' in b or 'kitchen' in b or 'counter' in b) and abs(p[0]-dp[0])<1.0 and abs(p[1]-dp[1])<0.8:
        print(" ",gn(g),"body",b,"type",m.geom_type[g],"size",np.round(m.geom_size[g],3),"pos",np.round(p,3),"ct",m.geom_contype[g],m.geom_conaffinity[g],"rbound",round(m.geom_rbound[g],3))
act=np.zeros(env.action_space.shape)
for t in range(40):
    env.step(act)
    if t in (0,2,5,10,20,39):
        print("step",t,"drink",np.round(d.xpos[bid],3),np.round(d.xquat[bid],3)); contacts()
for j in range(m.njnt):
    n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
    if n and ('drawer' in n): print("joint end",n,"qpos",d.qpos[m.jnt_qposadr[j]])

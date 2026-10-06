import sys, numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import GeneratedApproach
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
ap = GeneratedApproach(env.action_space, env.observation_space, {})
seed=int(sys.argv[1]); obs,info=env.reset(seed=seed); ap.reset(obs,info)
re = env._kinder_env._object_centric_env._robot_env; m, d = re.sim.model.mj_model, re.sim.data.mj_data
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site")
def contacts():
    out=set()
    for i in range(d.ncon):
        c=d.contact[i]; b1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom1]); b2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom2])
        if 'robot' in b1 or 'robot' in b2: out.add((b1,b2,tuple(np.round(c.pos,3))))
    return out
for k in range(200):
    a=ap.get_action(obs)
    if ap.phase in ("descend",) and k%5==0:
        tp,_=ap._fk(ap.base_target, ap.q_target)
        print(k, ap.phase, "pinch",d.site_xpos[sid].round(3),"tgt",tp.round(3),"err",round(ap._arm_err(),3),"drink",obs[16:19].round(3), contacts())
    obs,r,t,_,_=env.step(a)
    if ap.phase=="close": break

from run_test import *
import mujoco, approach as A
s, info = env.reset(seed=0, options={'object_count': 5})
oc = env._current_backend._object_centric_env
m = oc._robot_env.sim.model.mj_model; d = oc._robot_env.sim.data.mj_data
site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'robot_pinch_site')
wb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'wiper_0')
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s, info)
for t in range(220):
    a = ap.get_action(s); s,*_ = env.step(a)
    if t>=88 and t%5==0:
        R = d.site_xmat[site].reshape(3,3); Rw = d.xmat[wb].reshape(3,3)
        # contacts with wiper
        cs=[]
        for i in range(d.ncon):
            c=d.contact[i]; b1=m.geom_bodyid[c.geom1]; b2=m.geom_bodyid[c.geom2]
            if wb in (b1,b2):
                o = b2 if b1==wb else b1
                cs.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, o))
        print(t, len(ap.queue), 'grip z-axis', R[:,2].round(2), 'wiper z-axis', Rw[:,2].round(2), 'gripq', d.qpos[62].round(3), 'contacts', sorted(set(cs)))

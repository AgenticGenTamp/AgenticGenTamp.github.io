from run_test import *
import mujoco, approach as A
s, info = env.reset(seed=0, options={'object_count': 5})
oc = env._current_backend._object_centric_env
m = oc._robot_env.sim.model.mj_model; d = oc._robot_env.sim.data.mj_data
site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'robot_pinch_site')
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s, info)
for t in range(130):
    a = ap.get_action(s); s,*_ = env.step(a)
    b, arm, g, cubes, w = ap._read(s)
    if t>=60 and t%3==0:
        tg = ap.queue[0] if ap.queue else None
        wq = s.get(s.get_object_from_name('wiper_0'),'qx'), s.get(s.get_object_from_name('wiper_0'),'qy')
        print(t, len(ap.queue), 'base', b.round(3), 'pinch', d.site_xpos[site].round(3), 'wiper', w.round(3), 'tilt', np.round(wq,3), 'cubes y', np.round(sorted([c[1] for c in cubes]),2))

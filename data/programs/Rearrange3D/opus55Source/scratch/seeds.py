import time, numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
oc = env._kinder_env._object_centric_env
def summ(st):
    out={}
    for o in st:
        if o.name in ("bowl_0","boxed_drink_0","can_0"):
            out[o.name]=np.round([st.get(o,f) for f in ("x","y","z","qz")],3)
        if o.name=="robot":
            out["base"]=np.round([st.get(o,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")],3)
            out["arm"]=np.round([st.get(o,f"pos_arm_joint{i}") for i in range(1,8)],3)
            out["grip"]=round(float(st.get(o,"pos_gripper")),3)
    return out
for s in range(5):
    t=time.time(); obs,_=env.reset(seed=s); rt=time.time()-t
    print("seed",s,"reset %.2fs"%rt, summ(oc._get_current_state()))
re = oc._robot_env
m, d = re.sim.model.mj_model, re.sim.data.mj_data
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site")
print("pinch site", np.round(d.site_xpos[sid],3), np.round(d.site_xmat[sid].reshape(3,3),2).tolist())
obs,_=env.reset(seed=0)
a=np.zeros(11); t=time.time()
for i in range(20): obs,r,term,trunc,info=env.step(a)
print("step time %.4fs"%((time.time()-t)/20), "reward",r,term,trunc)
print("after 20 zero steps", summ(oc._get_current_state()))
# delta tracking test
q0=summ(oc._get_current_state())["arm"]
a=np.zeros(11); a[3]=0.1; a[4]=0.05
for i in range(5):
    env.step(a); print("step",i, summ(oc._get_current_state())["arm"][:2], summ(oc._get_current_state())["base"])
for i in range(5):
    env.step(np.zeros(11)); print("hold",i, summ(oc._get_current_state())["arm"][:2])
a=np.zeros(11); a[0]=0.1
for i in range(3): env.step(a); print("base", summ(oc._get_current_state())["base"])
a=np.zeros(11); a[10]=1
for i in range(5): env.step(a); print("grip", summ(oc._get_current_state())["grip"])
print("obs vector len", len(obs), "obs[:20]", np.round(obs[:20],3))

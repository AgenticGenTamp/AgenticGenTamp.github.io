import time, numpy as np
np.set_printoptions(precision=4, suppress=True)
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env=KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
oc=env._kinder_env._object_centric_env
re=oc._robot_env
for s in range(5):
    obs,_=env.reset(seed=s)
    ss=oc._objects_dict['seesaw_1']
    print(f"seed{s} LB {obs[0:3]} q{obs[3:7]} bb{obs[13:16]} | S1 {obs[54:57]} | S2 {obs[70:73]} | seesaw {obs[38:41]} q{obs[41:45]} bb{obs[51:54]} tilt={ss.get_beam_tilt_angle_degrees():.2f}deg")
    print(f"   robot base {obs[16:19]} arm {obs[19:26]} grip {obs[26]:.3f}")
    import mujoco
    m=re.sim.model.mj_model; d=re.sim.data.mj_data
    if s==0:
        for n in ['large_block','small_block_1','seesaw_1_beam','seesaw_1']:
            bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n); print("  body",n,d.xpos[bid], "mass",m.body_mass[bid])
        for g in range(m.ngeom):
            nm=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g)
            if nm and ('block' in nm or 'seesaw' in nm): print("  geom",nm,m.geom_size[g], m.geom_type[g], m.geom_friction[g])
        for i in range(m.nsite):
            nm=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SITE,i)
            if nm and ('goal' in nm or 'beam' in nm): print("  site",nm,d.site_xpos[i],m.site_size[i], "parent", mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.site_bodyid[i]))
        print("  region bbox", ss.region_objects if hasattr(ss,'region_objects') else None)
        for rn,rl in ss.region_objects.items():
            for r in rl: print("   ",rn, np.round(r.bbox,4))
        print(" opt.timestep", m.opt.timestep, "ctrl names", list(re.ctrl.keys()), {k:len(v) for k,v in re.ctrl.items()})
        print(" ctrlrange base", m.actuator_ctrlrange[:3])
# timing
obs,_=env.reset(seed=0)
a=np.zeros(11,dtype=np.float32)
ts=[]
for i in range(20):
    t=time.time(); obs,r,term,trunc,info=env.step(a); ts.append(time.time()-t)
print("step time mean %.4f min %.4f"%(np.mean(ts),np.min(ts)), "reward",r,term,trunc)
st=env.get_state()
ts=[]
for i in range(5):
    t=time.time(); env.set_state(st); ts.append(time.time()-t)
print("set_state time %.4f"%np.mean(ts))
t=time.time(); g=env.get_state(); print("get_state", time.time()-t)
d=env.get_state()-st; print("set_state roundtrip max diff", np.abs(d).max(), np.argmax(np.abs(d)))
# determinism after set_state
env.set_state(st); o1=[env.step(a)[0] for _ in range(5)][-1]
env.set_state(st); o2=[env.step(a)[0] for _ in range(5)][-1]
print("replay diff after set_state", np.abs(o1-o2).max())

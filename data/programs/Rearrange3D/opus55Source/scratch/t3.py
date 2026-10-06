import sys; sys.path.insert(0,'/sandbox/scratch'); from h import *
h=H(int(sys.argv[1])); h.step(60); m,d=h.m,h.d
for name in ["bowl_0","boxed_drink_0","can_0"]:
    b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
    print(name, np.round(d.xpos[b],3), "jnt", m.body_jntadr[b], "qadr", m.jnt_qposadr[m.body_jntadr[b]])
    for g in range(m.ngeom):
        if m.geom_bodyid[g]==b: print("  geom",m.geom_type[g],np.round(m.geom_size[g],3),np.round(m.geom_pos[g],3), "aabb", np.round(m.geom_aabb[g],3))
print("obs", np.round(h.obs[:60],3))

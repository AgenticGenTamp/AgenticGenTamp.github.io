import numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
k = env._kinder_env
oc = k._object_centric_env
obs, _ = env.reset(seed=0)
re = oc._robot_env; sim = re.sim
m = sim.model.mj_model; d = sim.data.mj_data
print("timestep", m.opt.timestep, "neq", m.neq, "nu", m.nu, "gravity", m.opt.gravity)
for i in range(m.neq): print("eq", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i), m.eq_type[i], m.eq_active0[i])
def dump(tag):
    print("==", tag)
    for dn in ["s1c1"]:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"kitchen_island_drawer_{dn}")
        print("body", dn, np.round(d.xpos[b],4))
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in (b,) or m.body_parentid[m.geom_bodyid[g]]==b:
                gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                R = d.geom_xmat[g].reshape(3,3); half = np.abs(R) @ m.geom_size[g]
                lo = d.geom_xpos[g]-half; hi = d.geom_xpos[g]+half
                print("  geom", gn, "lo", np.round(lo,4), "hi", np.round(hi,4), "fric", m.geom_friction[g])
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"kitchen_island_drawer_{dn}_joint")
        print("  joint range", m.jnt_range[j], "axis(local)", m.jnt_axis[j], "damping", m.dof_damping[m.jnt_dofadr[j]])
    isl = oc._fixtures_dict["kitchen_island"]
    for r in isl.region_objects["kitchen_island_shelf_1_partition_1_region"]: print("  target bbox", np.round(r.bbox,4))
for dn in ["s0c0","s0c1","s0c2","s1c0","s1c1","s1c2"]:
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"kitchen_island_drawer_{dn}")
    h = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"kitchen_island_drawer_{dn}_handle")
    print(dn, "body", np.round(d.xpos[b],4), "handle", np.round(d.xpos[h],4))
# island geoms
for g in range(m.ngeom):
    gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if gn.startswith("kitchen_island") and "drawer" not in gn:
        R = d.geom_xmat[g].reshape(3,3); half = np.abs(R) @ m.geom_size[g]
        print(" isl geom", gn, np.round(d.geom_xpos[g]-half,4), np.round(d.geom_xpos[g]+half,4))
dump("closed")
j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "kitchen_island_drawer_s1c1_joint")
d.qpos[m.jnt_qposadr[j]] = 0.4; mujoco.mj_forward(m, d)
dump("pos=0.4")
# wiper and cube geoms
for nm in ["wiper_0","cube_0"]:
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
    print(nm, "mass", m.body_mass[b], "pos", np.round(d.xpos[b],4))
    for g in range(m.ngeom):
        if m.geom_bodyid[g]==b:
            R = d.geom_xmat[g].reshape(3,3); half = np.abs(R) @ m.geom_size[g]
            print("  ", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), "type", m.geom_type[g], "size", m.geom_size[g], "lo", np.round(d.geom_xpos[g]-half,4), "hi", np.round(d.geom_xpos[g]+half,4), "fric", m.geom_friction[g])
print("sites:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite) if 'shelf_1_partition_1' in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or '')])
r = oc._fixtures_dict["kitchen_island"].region_objects["kitchen_island_shelf_1_partition_1_region"][0]
print("site name", r.site_element.get("name"), "env", r.env, hasattr(re,'sim'))
try:
    print(sim.data.get_site_xpos(r.site_element.get("name")))
except Exception as e: print("ERR", repr(e))
for i in range(m.neq):
    print("eq", m.eq_type[i], mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.eq_obj1id[i]) if m.eq_type[i]!=2 else mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.eq_obj1id[i]))
r.env = re
print("dyn bbox pos=0.4", np.round(r.bbox,4))
d.qpos[m.jnt_qposadr[j]] = 0.0; mujoco.mj_forward(m, d)
print("dyn bbox pos=0", np.round(r.bbox,4))
r.env=None

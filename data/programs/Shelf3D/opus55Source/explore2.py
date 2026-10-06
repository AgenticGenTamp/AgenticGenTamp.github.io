import numpy as np, mujoco
m=mujoco.MjModel.from_xml_path("/robocode/third-party/kindergarden/src/kinder/envs/dynamic3d/models/stanford_tidybot/tidybot.xml")
d=mujoco.MjData(m)
print("nq",m.nq,[m.joint(i).name for i in range(m.njnt)])
q0=[0,-0.349,3.142,-2.548,0,-0.873,1.571]
d.qpos[3:10]=q0; mujoco.mj_forward(m,d)
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"pinch_site")
print("pinch",d.site_xpos[sid], d.site_xmat[sid].reshape(3,3))
# geoms: print bounding of collision geoms in world (base at 0)
for g in range(m.ngeom):
    if m.geom_contype[g]==0 and m.geom_conaffinity[g]==0: continue
    b=m.body(m.geom_bodyid[g]).name
    print(g,b,m.geom_type[g],np.round(d.geom_xpos[g],3),np.round(m.geom_size[g],3), m.geom_rbound[g].round(3))

import sys; sys.path.insert(0,'/sandbox/scratch'); from h import *
from scipy.spatial.transform import Rotation as R
oc=env._kinder_env._object_centric_env
h=H(32); h.step(60)
bowl=oc._objects_dict['bowl_0']; reg=bowl.region_objects['bowl_0_region'][0]
m,d=h.m,h.d
jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"bowl_0_joint") if mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,"bowl_0_joint")>=0 else None
bb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"bowl_0"); ja=m.body_jntadr[bb]; qa=m.jnt_qposadr[ja]
print("bowl qpos",d.qpos[qa:qa+7]); sid=reg.site_element.get("name"); print("site",sid, reg.site_element.attrib, re.sim.data.get_site_xpos(sid))
q0=d.qpos.copy()
for th in [0,10,15,20,25,30]:
    d.qpos[qa:qa+3]=[0.3,0.0,0.478]; quat=R.from_euler('y',th,degrees=True).as_quat(); d.qpos[qa+3:qa+7]=[quat[3],*quat[:3]]
    mujoco.mj_forward(m,d); b=np.array(reg.bbox); sp=re.sim.data.get_site_xpos(sid)
    print(th, "site",np.round(sp,3),"bbox",np.round(b,3))
d.qpos[:]=q0; mujoco.mj_forward(m,d)

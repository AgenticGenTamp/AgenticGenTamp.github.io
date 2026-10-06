import numpy as np, mujoco
from run_test import env
s,info=env.reset(seed=0,options={'object_count':1})
oc=env._current_backend._object_centric_env; m=oc._robot_env.sim.model.mj_model; d=oc._robot_env.sim.data.mj_data
pads=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'robot_'+g) for g in ['left_pad1','right_pad1']]
for g in [0,0.3,0.4,0.5,0.6,0.7,0.0]:
    for k in range(6):
        a=np.zeros(11,np.float32); a[10]=g; s,*_=env.step(a)
        print(g,k,round(float(np.linalg.norm(d.geom_xpos[pads[0]]-d.geom_xpos[pads[1]])),4))

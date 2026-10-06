import numpy as np, mujoco
from test_approach import make_env
env=make_env()
s,_=env.reset(seed=4, options={'object_count':2})
oc=env._current_backend._object_centric_env
re=oc._robot_env
m,d=re.sim.model.mj_model, re.sim.data.mj_data
o=s.get_object_from_name('cupboard_1'); print('cup1', s.get(o,'x'), s.get(o,'y'))
for g in range(m.ngeom):
    b=m.body(m.geom_bodyid[g]).name
    if 'cupboard_1' in b or 'cupboard_1' in m.geom(g).name:
        R=d.geom_xmat[g].reshape(3,3); sz=m.geom_size[g]
        ext=np.abs(R)@sz
        print(m.geom(g).name, m.geom_type[g], np.round(d.geom_xpos[g]-ext,3), np.round(d.geom_xpos[g]+ext,3), m.geom_contype[g], m.geom_conaffinity[g])

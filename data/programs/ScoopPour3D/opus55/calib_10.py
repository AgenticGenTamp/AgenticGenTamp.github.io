import numpy as np, json
from calib_util import *
r=R()
r.goto_base([-0.14,0.0,0.0])
q,_=r.ik_world(np.array([0.2,-0.2,0.52]),tool=-0.224); r.goto_q(q,settle=10)
r.gripper(0.0,20)
v=np.concatenate([r.obs.data[o] for o in sorted(r.obs.data, key=lambda o:o.name)])
json.dump([float(x) for x in np.asarray(v)],open('calib_vec.json','w'))
print(len(v))

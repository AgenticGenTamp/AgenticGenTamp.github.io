import numpy as np
from approach import Planner, _Info, _yaw_quat
pl = Planner(1, 1e18)
s,_ = pl.sim.reset(seed=0)
for bx in [-0.2, -0.25]:
    st = s.copy(); r = st.get_object_from_name('robot'); st.set(r, 'pos_base_x', bx)
    pl.set(st)
    for x in [0.11, 0.3, 0.49]:
      for y in [-0.38, -0.2, 0.2, 0.38]:
        row=[]
        for yaw in [0.0, np.pi/2, np.pi/4]:
            q = pl.ik([x, y, 0.25], _yaw_quat(yaw), _Info(st).joints())
            if q is None: row.append('hoverfail'); continue
            fake = pl.with_joints(st, q)
            acts = pl.move_actions(fake, [x, y, 0.105], _yaw_quat(yaw))
            row.append('ok' if acts is not None else 'descfail')
        print('R', bx, x, y, row)

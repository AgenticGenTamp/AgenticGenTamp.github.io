import numpy as np
from approach import Planner, _Info, _yaw_quat, _coerce
from kinder.envs.kinematic3d.obstruction3d import ObjectCentricObstruction3DEnv
pl = Planner(1, 1e18)
s,_ = pl.sim.reset(seed=0)
for bx in [-0.12, -0.2, -0.3]:
    st = s.copy(); r = st.get_object_from_name('robot'); st.set(r, 'pos_base_x', bx)
    pl.set(st)
    for x in [0.10, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5]:
        row=[]
        for yaw in [0.0, np.pi/2]:
            q = pl.ik([x, 0.0, 0.25], _yaw_quat(yaw), _Info(st).joints())
            if q is None: row.append('hoverfail'); continue
            fake = pl.with_joints(st, q)
            acts = pl.move_actions(fake, [x, 0.0, 0.11], _yaw_quat(yaw))
            row.append('ok' if acts is not None else 'descfail')
        print(bx, x, row)

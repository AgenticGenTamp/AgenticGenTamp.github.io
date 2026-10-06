import sys, numpy as np
TT=int(sys.argv[3]); sys.argv=['x',sys.argv[1],sys.argv[2],'0']
exec(open('dbg.py').read().split("for i in range(T)")[0])
oc = env._current_backend._object_centric_env
sim = oc._robot_env.sim; m=sim.model.mj_model; d=sim.data.mj_data
for i in range(TT):
    a=ap.get_action(s); s,*_=env.step(a)
b,arm,g=ap._robot(s); ee=ap._ee(b,arm); print('ee',ee.round(3), 'cube', ap._pose(s,ap.cur)[:3].round(3))
for gi in range(m.ngeom):
    n=m.geom(gi).name or m.body(m.geom_bodyid[gi]).name
    if 'pad' in n or 'follower' in n or 'coupler' in n:
        R=d.geom_xmat[gi].reshape(3,3); hs=m.geom_size[gi]
        if m.geom_type[gi]==6:
            corners=np.array([[sx*hs[0],sy*hs[1],sz*hs[2]] for sx in(-1,1) for sy in(-1,1) for sz in(-1,1)])@R.T+d.geom_xpos[gi]
            print(n, 'type',m.geom_type[gi],'min',corners.min(0).round(3),'max',corners.max(0).round(3))
        else:
            print(n,'type',m.geom_type[gi],'pos',d.geom_xpos[gi].round(3),'size',hs.round(3))

import sys
TT=int(sys.argv[3]); sys.argv=['x',sys.argv[1],sys.argv[2],'0']
exec(open('dbg.py').read().split("for i in range(T)")[0])
oc = env._current_backend._object_centric_env
sim = oc._robot_env.sim; m=sim.model.mj_model; d=sim.data.mj_data
nm=lambda g: m.geom(g).name or m.body(m.geom_bodyid[g]).name
for i in range(TT):
    a=ap.get_action(s); s,*_=env.step(a)
    if ap.phase in ('approach','descend','close','pdown','pmove'):
        b,arm,g=ap._robot(s); ee=ap._ee(b,arm)
        con=set(nm(d.contact[k].geom1)+'|'+nm(d.contact[k].geom2) for k in range(d.ncon))
#        print('POS',[(nm(d.contact[k].geom1),d.contact[k].pos.round(3)) for k in range(d.ncon) if 'bin_green' in nm(d.contact[k].geom1)+nm(d.contact[k].geom2)])
        con=[c for c in con if 'robot' in c]
        print(i, ap.phase, ap.cur, 'yaw',round(ap.yaw,2),'ee',ee.round(3), {c:ap._pose(s,c)[:2].round(3).tolist() for c in ap.cubes if np.linalg.norm(ap._pose(s,c)[:2]-ee[:2])<0.06}, con)

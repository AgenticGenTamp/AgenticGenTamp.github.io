import numpy as np, kin
p,R = kin.fk(0,0,0,kin.Q_HOME); print('home tip', p.round(3), 'axis', R[:,2].round(2), 'close', R[:,1].round(2))
for d in [0.4,0.5,0.55,0.6,0.7]:
    for z in [0.06, 0.012]:
        q,ok,e = kin.ik_multi((0,0,0), kin.Q_HOME, [d,0,z], 'down', yaw=np.pi/2, yaw_sym=np.pi/2)
        print(d, z, ok, np.round(q,3), 'maxdiff', np.abs(q-kin.Q_HOME).max().round(2))

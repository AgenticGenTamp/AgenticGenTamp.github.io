import numpy as np, approach as ap, time
K=ap._Kin.get()
for g in np.deg2rad([-90,-60,-30,0,30,60,90]):
    for z in [0.035,0.155,0.3]:
        t=time.time(); q=K.canon(z,ap.TILT,g); dt=time.time()-t
        p,R=K.fk(q); print(round(np.rad2deg(g)), z, np.round(p,3), 'zaxis',np.round(R[:,2],2),'xaxis',np.round(R[:,0],2), f"{dt*1000:.0f}ms", np.round(q,2))

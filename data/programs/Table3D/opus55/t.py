from fk import *
np.set_printoptions(precision=3,suppress=True)
for j2 in [0.65,0.75]:
    q=[0,j2,-np.pi,-2.5,0,-0.87,np.pi/2]
    T,fr=fk_arm(q,0.12); print(j2, T[:3,3], [f[:3,3] for f in fr[4:]])

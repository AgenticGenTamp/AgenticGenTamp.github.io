from kin import *
np.set_printoptions(precision=3,suppress=True)
q0=np.array([0,-0.35,-np.pi,-2.5,0,-0.87,np.pi/2])
T=fk(q0); print(T)
q,e=ik(q0, np.array([0.4,0.1,0.0]), down_R(0)); print(q,e); print(fk(q))

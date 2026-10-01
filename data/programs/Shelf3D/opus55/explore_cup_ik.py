from explore_cup_lib import *
q=np.array([-0.07,0.01,3.21,-2.32,0.01,0.76,1.57])  # horizontal pose, cube (0.669,0,0.88)
np.set_printoptions(precision=2,suppress=True)
for z in [0.88,0.8,0.7,0.6,0.5,0.4,0.35,0.3,0.2,0.1,0.07]:
    q2,e=ikc(np.array([0.60,0,z]),Rh,q); print(z,round(e,4),q2, np.abs(q2-q).max().round(2)); q=q2
for x in [0.5,0.4,0.3]:
    q2,e=ikc(np.array([x,0,0.07]),Rh,q); print(x,round(e,4),q2, np.abs(q2-q).max().round(2)); q=q2

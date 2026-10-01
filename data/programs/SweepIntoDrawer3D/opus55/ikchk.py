import numpy as np
from kin import ik, R_down, fk_world
Q_HOME = np.array([0, -0.349, 3.142, -2.548, 0, -0.873, 1.571])
s=np.array(eval(open('/tmp/s.txt').read()))
for B in ([1.25,-0.7,np.pi],[1.25,-0.55,np.pi],[1.15,-0.6,np.pi],[1.35,-0.6,np.pi]):
    B=np.array(B)
    for i in range(5):
        c=s[16*i:16*i+3]; yaw=2*np.arctan2(s[16*i+6],s[16*i+3])
        out=[]
        for k in range(2):
            gy=yaw+k*np.pi/2
            gy=min([gy,gy+np.pi,gy-np.pi],key=lambda a:abs(((a-np.pi)+np.pi)%(2*np.pi)-np.pi))
            q,e=ik(B,Q_HOME,np.array([c[0],c[1],0.462]),R_down(gy),iters=300)
            out.append(f"{e:.4f}")
        print(B[:2],i,np.round(c[:2],3),out)

import numpy as np
from scipy.spatial.transform import Rotation as R
q=[.67717,-.34313,1.2,-1.46688,1.24223,-1.95443,2.22254]
p=np.array([0.,.188,0.]); r=np.eye(3)
for a,axis,offset in zip(q,['z','y','x','y','x','y','x'],[[0,0,0],[.1,0,0],[0,0,0],[.4,0,0],[0,0,0],[.321,0,0],[0,0,0]]):
 p+=r@offset; r=r@R.from_euler(axis,a).as_matrix()
p+=r@np.array([.18,0,0])
print(p,r, R.from_matrix(r).as_quat())

import numpy as np, sys
import throw_test as T
from common import *
bt=T.grasp(int(sys.argv[1]),dz=float(sys.argv[2]))
s=T.S['s']; print('after grasp cube', obj(s,'cube_0')[:3].round(3), 'pinch', T.k.fk(*robot(s))[0].round(3))
base,q=robot(s)
for qt in [q+np.array([0,-0.5,0,0,0,0,0]), q+np.array([0,-0.5,0,0.5,0,0,0]), np.array([0,-0.5,np.pi,-1.5,0,-1.0,1.57])]:
    T.slow(bt,qt,1.0,0.5)
    s=T.S['s']; print('cube', obj(s,'cube_0')[:3].round(3), 'pinch', T.k.fk(*robot(s))[0].round(3))

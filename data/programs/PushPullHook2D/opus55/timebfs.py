import time,numpy as np
from approach import *
valid=np.ones((67,22,64),bool); valid[30:35,:15,:]=False
t=time.time(); d=bfs(valid,(0,0,0),CONN3); print('bfs3',time.time()-t, d.max())

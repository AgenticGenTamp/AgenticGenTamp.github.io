import sys,time; sys.path.insert(0,'.')
from gridroute import grid_path
from approach import base_ok, GeneratedApproach
import numpy as np
ap=GeneratedApproach(None,None,{}); ap.bcache={}
for b0,b1 in [((-1,0,0),(-0.3,-0.99,0)),((-0.53,-0.02,0),(0.2,-0.99,0)),((-0.53,0,0),(0.6,0,3.14159)),((-0.9,0,0),(0.3,1.0,-1.57))]:
    t=time.time(); p=grid_path(b0,b1,base_ok); dt=time.time()-t
    r=ap._base_graph(np.array(b0,float),np.array(b1,float))
    print(len(p) if p else None, "graph", r[3] if r else None, "%.3fs"%dt, [tuple(np.round(q,2)) for q in (p or [])][:8])

import numpy as np, json
import kin
from probe_lib import *
d=json.load(open('calib.json')); base=np.array(d['base'])
mx=0
for r in d['rows']:
    q=np.array(r['q']); p=kin.linpos(q)+np.array([base[0],base[1],0.0])
    mx=max(mx,np.linalg.norm(p-np.array(r['part'][:3])))
print("model max err on calib data",round(mx,6))
q0=np.array([0.0,-0.35,-3.1416,-2.5,0.0,-0.87,1.5708])
q,p,R=kin.ikin(q0,np.array([0.3,0.2,0.2]),Rdown)
print("ik check pos",np.round(p,5),"Rerr",np.round(np.linalg.norm(R-Rdown),6))

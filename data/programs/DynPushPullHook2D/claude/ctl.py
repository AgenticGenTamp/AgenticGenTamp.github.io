import numpy as np
MX=[0.05,0.05,0.06544985,0.1,0.02]
def act(dx=0,dy=0,dth=0,da=0,dg=0):
    a=np.array([dx,dy,dth,da,dg],dtype=np.float64)
    lo=np.array(MX); a=np.clip(a,-lo*0.995,lo*0.995)
    return a.astype(np.float32)
def rget(obs,f):
    return float(obs.get(obs.get_object_from_name("robot"),f))
def oget(obs,name,f):
    return float(obs.get(obs.get_object_from_name(name),f))

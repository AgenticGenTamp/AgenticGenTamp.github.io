import numpy as np, json
from env_client import make_env
from probe_sweep_lib import *
np.set_printoptions(precision=4,suppress=True,linewidth=200)
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
c0=cubes(o)
print("base",np.round(o[125:128],4),"grip",o[135])
print("cubes0\n",np.round(c0,4))
print("wiper",np.round(wiper(o),4))
ymid=float(np.mean(c0[:,1]))
print("ymid",round(ymid,4))
# approach: high, behind pile
o,d=movew(env,o,(0.58,ymid,0.62),steps=150,grip=0.0); print("A",d)
o,d=movew(env,o,(0.58,ymid,0.475),steps=150,grip=0.0); print("B",d)
rows=[]
x=0.58
while x<1.06:
    x+=0.02
    o,d=movew(env,o,(x,ymid,0.475),steps=60,grip=0.0)
    c=cubes(o)
    rows.append((round(x,3),d['err'],d['ikres'],d['used'],np.round(c,3).tolist()))
    print(f"x={x:.2f} err={d['err']:.4f} ikres={d['ikres']:.5f} u={d['used']} cubes={np.round(c,3).tolist()}")
print("FINAL cubes\n",np.round(cubes(o),4))
print("wiper",np.round(wiper(o),4))
json.dump(rows,open("res_sweep1.json","w"))
env.close()

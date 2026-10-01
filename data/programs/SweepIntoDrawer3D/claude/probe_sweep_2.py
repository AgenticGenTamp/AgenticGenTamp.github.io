import numpy as np, json
from env_client import make_env
from probe_sweep_lib import *
np.set_printoptions(precision=4,suppress=True,linewidth=200)
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
print("base",np.round(o[125:128],4))
print("cubes0\n",np.round(cubes(o),4))
# --- Part A: vertical contact probe on empty counter spot (closed gripper)
o,d=movew(env,o,(0.78,-0.28,0.60),steps=150,grip=0.0); print("preA",d)
for z in np.arange(0.52,0.395,-0.01):
    o,d=movew(env,o,(0.78,-0.28,float(z)),steps=80,grip=0.0)
    print(f"A z={z:.3f} err={d['err']:.4f} u={d['used']} ikres={d['ikres']:.5f}")
o,d=movew(env,o,(0.78,-0.28,0.60),steps=120,grip=0.0)
# same with open gripper
for z in [0.50,0.48,0.47,0.46,0.45,0.44,0.43]:
    o,d=movew(env,o,(0.78,-0.28,float(z)),steps=80,grip=1.0)
    print(f"Aopen z={z:.3f} err={d['err']:.4f} u={d['used']}")
o,d=movew(env,o,(0.78,-0.28,0.60),steps=120,grip=1.0)
print("wiper after A",np.round(wiper(o),4))
# --- Part B: base-drag sweep with open gripper
c0=cubes(o); ymid=float(np.mean(c0[:,1])); print("ymid",ymid)
o,d=movew(env,o,(0.60,ymid,0.62),steps=150,grip=1.0); print("B0",d)
o,d=movew(env,o,(0.60,ymid,0.478),steps=150,grip=1.0); print("B1",d)
qhold=o[128:135].copy()
bx0,by0,yaw0=o[125],o[126],o[127]
rows=[]
for i in range(1,15):
    tgt=[bx0+0.03*i,by0,yaw0]
    o=move_base(env,o,tgt,steps=8,grip=1.0,qhold=qhold)
    c=cubes(o)
    print(f"drag i={i} base={np.round(o[125:128],3)} qerr={np.abs(qhold-o[128:135]).max():.4f} cubes={np.round(c,3).tolist()}")
    rows.append((float(o[125]),np.round(c,4).tolist()))
print("FINAL cubes\n",np.round(cubes(o),4))
print("wiper",np.round(wiper(o),4))
json.dump(rows,open("res_sweep2.json","w"))
env.close()

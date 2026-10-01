import numpy as np, json
from env_client import make_env
from probe_sweep_lib import *
np.set_printoptions(precision=3,suppress=True,linewidth=200)
env=make_env(); o,_=env.reset(seed=0); o=np.asarray(o,float)
HOME=[float(o[125]),float(o[126]),float(o[127])]
TOT=[0]
def cnt(o): return o
def summ(tag,o):
    c=cubes(o); print(tag,"steps~",TOT[0],"cubes",np.round(c,3).tolist(),"wiper",np.round(wiper(o),3).tolist())
def steps_of(d): TOT[0]+=d['used']
o0=o
print("base",np.round(o[125:128],3)); summ("init",o)
GRIP=0.0; ZS=0.470
def do_pass(o,y,dx=0.32,label=""):
    o,d=movew(env,o,(0.62,y,0.60),steps=100,grip=GRIP); steps_of(d); e1=d['err']
    o,d=movew(env,o,(0.62,y,ZS),steps=100,grip=GRIP); steps_of(d); e2=d['err']
    q=o[128:135].copy(); bx0=o[125]
    n0=TOT[0]
    for k in range(1,5):
        o=move_base(env,o,[bx0+dx*k/4,o0[126],o0[127]],steps=6,grip=GRIP,qhold=q); TOT[0]+=6
        c=cubes(o); print(f"  {label} bx={o[125]:.3f} c={np.round(c,3).tolist()}")
    o,d=movew(env,o,(o[125]-0.62+0.62,y,0.62),steps=60,grip=GRIP)  # lift in place (local)
    steps_of(d)
    o=move_base(env,o,HOME,steps=12,grip=GRIP); TOT[0]+=12
    print(f"  {label} e1={e1:.3f} e2={e2:.3f} tot={TOT[0]}")
    return o
ys=[-0.020,-0.063,-0.136]
for i,y in enumerate(ys):
    c=cubes(o); print(f"PASS{i} y={y} remaining_on_counter={[j for j in range(5) if c[j][2]>0.4]}")
    o=do_pass(o,y,label=f"P{i}")
summ("after 3 passes",o)
# extra: long drag on whatever remains near edge
c=cubes(o)
rem=[j for j in range(5) if c[j][2]>0.40]
print("rem",rem)
if rem:
    j=rem[0]; y=float(c[j][1]); x=float(c[j][0])
    o,d=movew(env,o,(x-0.10,y,0.60),steps=80,grip=GRIP); steps_of(d)
    o,d=movew(env,o,(x-0.10,y,ZS),steps=80,grip=GRIP); steps_of(d)
    q=o[128:135].copy(); bx0=o[125]
    for k in range(1,7):
        o=move_base(env,o,[bx0+0.06*k,o0[126],o0[127]],steps=5,grip=GRIP); TOT[0]+=5
        print(f"  X bx={o[125]:.3f} c{j}={np.round(cubes(o)[j],3)} qerr={np.abs(q-o[128:135]).max():.3f}")
summ("final",o); print("TOTSTEPS",TOT[0])
env.close()

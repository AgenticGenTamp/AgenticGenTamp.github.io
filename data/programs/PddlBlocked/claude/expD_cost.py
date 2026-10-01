import numpy as np, math
from expD_lib import *; from expD_core import *
env = make_env()
class C:
    def __init__(s,e): s.e=e; s.n=0
    def __getattr__(s,k): return getattr(s.e,k)
    def step(s,a): s.n+=1; return s.e.step(a)
ce = C(env)
for seed in [1,2]:
    qh = home_q(env, seed)
    ap, obs, _ = remove_blocker(env, seed)
    g0 = ap._blocks(obs)["green0"]; d = np.array(ap.dir)
    bt = base_for(g0,d,0.90,-0.188,0.0)
    obs, ok, why, e = drive(ce, ap, obs, bt, qh); n_drive = ce.n
    R = grasp_R(math.atan2(d[1],d[0])); gp = grasp_pose(g0,d); pre = gp - d*0.22
    r = ap._robot(obs); q = try_ik(pre,R,r["base"],r["q"],seeds=6)
    obs,rej = servo(ce, ap, obs, bt, q, n=40); n_pre = ce.n - n_drive
    qq = ap._robot(obs)["q"]
    for k in range(1,12):
        p = pre + d*(0.02*k)
        qn = try_ik(p,R,bt,qq,seeds=3)
        obs,rej = servo(ce,ap,obs,bt,qn,n=6); qq = ap._robot(obs)["q"]
    n_ins = ce.n - n_drive - n_pre
    a=np.zeros(11,dtype=np.float32); a[10]=-1.0; obs,_,_,_,_=ce.step(a)
    ga = float(obs.get(obs.get_object_from_name("green0"),"grasp_active"))
    print(f"seed={seed} drive={n_drive} pregrasp={n_pre} insertion={n_ins} close=1 total={ce.n} G0={ga}")
    ce.n=0

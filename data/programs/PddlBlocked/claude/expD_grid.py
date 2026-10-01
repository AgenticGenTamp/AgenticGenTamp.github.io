import numpy as np, math, time, sys, json
from expD_lib import *
from expD_core import *
env = make_env()
seeds=[int(x) for x in sys.argv[1].split(",")]
LAT=[-0.188,-0.10,0.0,-0.30]; BACK=[0.70,0.78,0.85,0.90]; DYAW=[0.0,0.3,-0.3]
rows=[]
t0=time.time()
for seed in seeds:
    qh = home_q(env, seed)
    ap0, obs0, _ = remove_blocker(env, seed)
    d0 = np.array(ap0.dir); g00 = ap0._blocks(obs0)["green0"]
    print(f"### seed {seed} dir={np.round(d0,3)} yaw={math.atan2(d0[1],d0[0]):+.3f} g0={np.round(g00,3)}")
    print(f"{'lat':>6}{'back':>6}{'dyaw':>6} {'basex':>6}{'basey':>7} {'basewhy':>10} {'pre':>4}{'preerr':>8} {'gap':>7}{'lat_e':>7} {'stop':>6} {'gr':>3} {'st':>4}")
    for lat in LAT:
        for back in BACK:
            for dy in DYAW:
                ap, obs, _ = remove_blocker(env, seed)
                b = ap._blocks(obs); g0=b["green0"]; d=np.array(ap.dir)
                bt = base_for(g0,d,back,lat,dy)
                if max(abs(bt[0]),abs(bt[1]))>5.0:
                    print(f"{lat:6.3f}{back:6.2f}{dy:6.1f} {bt[0]:6.2f}{bt[1]:7.2f}  OUT_OF_BOUNDS")
                    rows.append(dict(seed=seed,lat=lat,back=back,dyaw=dy,base_why="oob")); continue
                obs,o = approach_grasp(env,ap,obs,g0,d,bt,qh,mode="direct")
                o.update(seed=seed,lat=lat,back=back,dyaw=dy,bx=float(bt[0]),by=float(bt[1]))
                rows.append(o)
                pe = -1 if o['pre_err'] is None else o['pre_err']
                gp = 99 if o['gap'] is None else o['gap']
                le = -1 if o['lat_err'] is None else o['lat_err']
                print(f"{lat:6.3f}{back:6.2f}{dy:6.1f} {bt[0]:6.2f}{bt[1]:7.2f} {o['base_why']:>10} {str(o['pre_ok'])[:1]:>4}{pe:8.3f} {gp:7.3f}{le:7.3f} {o['stop']:>6} {str(o['grasped'])[:1]:>3} {o['steps']:4d}")
    sys.stdout.flush()
json.dump(rows, open(f"expD_grid_{sys.argv[1].replace(',','_')}.json","w"), default=float)
print("t=%.1fs"%(time.time()-t0))

import numpy as np, math
from expD_lib import *; from expD_core import *
env = make_env()
for seed in [1,2]:
    qh = home_q(env, seed)
    for (bk,la,dy) in [(0.90,-0.188,0.0),(0.85,-0.188,0.3),(0.90,0.0,0.0)]:
        ap, obs, _ = remove_blocker(env, seed)
        g0 = ap._blocks(obs)["green0"]; d = np.array(ap.dir)
        bt = base_for(g0,d,bk,la,dy)
        obs,o = approach_grasp(env,ap,obs,g0,d,bt,qh,mode="direct")
        ga = float(obs.get(obs.get_object_from_name("green0"),"grasp_active"))
        print(f"seed={seed} back={bk} lat={la} dyaw={dy} why={o['base_why']} pre={o['pre_ok']} gap={o['gap']:.3f} stop={o['stop']} G0={ga} steps={o['steps']}")

import sys, math, time, numpy as np
sys.path.insert(0,'.')
from approach import *
from approach import _Rz, _Ry, _wrap
from robocode.environments.pr2_tamp_blocked_variable_count_env import PR2BlockedVariableCountEnv
env = PR2BlockedVariableCountEnv(design_counts=[0,1,2], eval_counts=[0,1,2])
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
st,_ = env.reset(seed=seed)
n = len([o for o in st.get_object_names() if str(o).startswith('green')])-1
sim = Sim(n)
sim.sync(st, set_walls=True)
e = sim.env
ssp = sim.ssp
with e.client():
    q0 = sim.get_conf()
    pp, pq = ssp.get_pose(e.penned)
    B = np.array(pp); yaw = math.atan2(2*(pq[3]*pq[2]), 1-2*pq[2]**2)
    print("pen", B, yaw)
    t=time.time()
    res=[]
    for k in range(4):
        psi = yaw + k*np.pi/2
        for pitch in [math.radians(a) for a in (24, 20, 15, 0)]:
            for (ox, oz) in [(0.02,-0.06),(0.03,-0.065),(0.0,-0.07),(0.035,-0.05), (0.02, 0.0)]:
                Rt = _Rz(psi) @ _Ry(pitch)
                T = B - Rt @ np.array([ox,0,oz])
                # base guess: tool 0.68 ahead
                for bo in (0.0,):
                    th = psi
                    bxy = T[:2] - 0.68*np.array([math.cos(th), math.sin(th)]) - 0.0
                    qi = q0.copy(); qi[0:2]=bxy; qi[2]=_wrap(th)
                    q, ok = sim.ik(T, Rt, qi, q_ref=q0)
                    if not ok: continue
                    sim.set_conf(q)
                    coll = e._in_collision()
                    cand = e._grasp_candidate()
                    res.append((n_steps(q0,q,sim.circ), k, round(math.degrees(pitch)), ox, oz, coll, cand==e.penned))
    print("time", time.time()-t)
    for r in sorted(res): print(r)
import pybullet as p
from pybullet_tools.utils import get_link_name, get_body_name
def diag(q):
    with e.client():
        sim.set_conf(q)
        out=[]
        for b in list(e.obstacles)+list(e.movables):
            pts = p.getClosestPoints(e.robot, b, 0.0, physicsClientId=e.client_id)
            for pt in pts:
                out.append((get_link_name(e.robot, pt[3]), b, round(pt[8],4)))
        return sorted(set(out))

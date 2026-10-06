exec(open('scratch/exp1.py').read().split("    t=time.time()")[0])
rng=np.random.default_rng(0)
import time
with e.client():
  t=time.time(); res=[]
  for trial in range(300):
    k = rng.integers(4); psi = yaw + k*np.pi/2
    pitch = math.radians(rng.uniform(10,25))
    ox = rng.uniform(0.0,0.04); oz = rng.uniform(-0.07,-0.03)
    Rt = _Rz(psi) @ _Ry(pitch)
    T = B - Rt @ np.array([ox,0,oz])
    r = rng.uniform(0.4,0.85); off = rng.uniform(-1.2,1.2)
    th = psi + rng.uniform(-0.6,0.6)
    bxy = T[:2] - r*np.array([math.cos(psi+off), math.sin(psi+off)])
    qi = q0.copy(); qi[0:2]=bxy; qi[2]=_wrap(th)
    qi[3:] = rng.uniform(sim.lo[3:], sim.hi[3:])
    q, ok = sim.ik(T, Rt, qi, q_ref=qi)
    if not ok: continue
    sim.set_conf(q)
    if e._in_collision(): continue
    if e._grasp_candidate()!=e.penned: continue
    res.append((n_steps(q0,q,sim.circ), int(k), round(math.degrees(pitch)), round(ox,3), round(oz,3), np.round(q,2)))
  print("time", time.time()-t, len(res))
  res.sort(key=lambda r:r[0])
  for r in res[:15]: print(r)

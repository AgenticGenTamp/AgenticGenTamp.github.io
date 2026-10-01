import numpy as np
from probe_base_lib import E

# --- rotation test at (-0.6,-0.8) ---
for sgn in [+1,-1]:
    e = E(0)
    oky = e.goto_y(-0.8); okx = e.goto_x(-0.6)
    print("at", np.round(e.r()[:3],3), "reached ok:", oky, okx)
    tgt = sgn*1.5; n=0; rejected=False
    while abs(e.r()[2]-tgt) > 1e-6 and n < 100:
        d = tgt - e.r()[2]
        rej = e.move_base(dr=np.clip(d,-0.05,0.05)); n+=1
        if rej:
            rejected=True; break
    print(f"  rot target {tgt:+.2f}: reached={np.round(e.r()[2],3)} rejected={rejected}")
    # after rotating, how far +x can we go?
    if not rejected:
        xr, rj = e.push_x(0.05, limit=1.6)
        xr, _ = (e.push_x(0.01, limit=1.6) if rj else (xr, None))
        print(f"  after rot: x_max={xr:+.3f} blocked={rj}")
    e.close()

# --- absolute travel limits far from table (y=-1.5) ---
e = E(0); e.goto_y(-1.5)
x, rej = e.push_x(0.05, limit=4.0, sign=+1)
print(f"y=-1.5 push +x: stopped at x={x:+.3f} blocked={rej}")
x, rej = e.push_x(0.05, limit=4.0, sign=-1)
print(f"y=-1.5 push -x: stopped at x={x:+.3f} blocked={rej}")
e.close()

# --- y travel limits at x=-1.0 ---
e = E(0)
n=0
while n<200:
    rej = e.move_base(dy=-0.05); n+=1
    if rej or e.r()[1] < -4.0: break
print(f"x=-1.0 push -y: stopped at y={e.r()[1]:+.3f} blocked={rej}")
e.close()

import numpy as np, fk, ctrl
q0=np.array([0.6772,-0.3431,1.2,-1.4669,1.2422,-1.9544,2.2225])
bx=-0.45
zs=0.80
print("feasible top-down targets in BASE frame (rel x fwd, rel y):")
for ry in np.arange(-0.6,0.81,0.1):
    row=""
    for rx in np.arange(0.2,1.01,0.05):
        tp=np.array([rx,ry,zs])
        best=False
        for ang in [np.pi/2, 0.0, np.pi/4]:
            for seed in [q0, np.array([0.3,-0.4,1.3,-0.7,1.5,-2.0,3.3])]:
                q,ok=fk.ik(tp,ctrl.targR(ang),seed,iters=120)
                if ok: best=True;break
            if best: break
        row+= "#" if best else "."
    print(f"{ry:+.1f} {row}")
print("cols rx from 0.20 step .05 to 1.00")

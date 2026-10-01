import numpy as np
from fk import fk
from ik import ik, LIM_LO, LIM_HI
from probe_fk_ctl import MOUNT, arm_from_world, RDOWN
np.set_printoptions(precision=3,suppress=True,linewidth=250)
# max radius from arm origin
best=0
rng=np.random.default_rng(0)
for _ in range(20000):
    q=rng.uniform(-np.pi,np.pi,7); q=np.clip(q,LIM_LO,LIM_HI)
    r=np.linalg.norm(fk(q)[:3,3]); best=max(best,r)
print("max |p_arm| sampled:",round(best,4))
print("straight-out reach fk([0,0,0,0,0,0,0]) z:",np.round(fk([0]*7)[:3,3],4))
q=np.zeros(7); q[1]=1.5708; print("arm horizontal fk:",np.round(fk(q)[:3,3],4), "r=",round(np.linalg.norm(fk(q)[:3,3]),4))
for base_x in [1.242,1.19]:
    base=np.array([base_x,-0.0837,np.pi])
    print(f"\n=== base x={base_x} yaw=pi -> arm origin world=({base_x-MOUNT[0]:.4f},{-0.0837-MOUNT[1]:.4f},{MOUNT[2]:.4f})")
    for z in [0.47,0.55,0.65]:
        row=[]
        for x in np.arange(0.40,1.05,0.05):
            ok=[]
            for y in np.arange(-0.55,0.30,0.05):
                pa=arm_from_world(base,[x,y,z])
                q,err=ik(pa,RDOWN,[0,-0.35,3.14,-2.55,0,-0.87,1.57])
                ok.append(err<0.005)
            row.append(f"{x:.2f}:{sum(ok)}")
        print(f" z={z}: reachable-y-count(of 17) "+" ".join(row))
    # y extent at z=0.5 for a few x
    for x in [0.55,0.65,0.75,0.85,0.95]:
        ys=[]
        for y in np.arange(-0.70,0.45,0.025):
            pa=arm_from_world(base,[x,y,0.50])
            q,err=ik(pa,RDOWN,[0,-0.35,3.14,-2.55,0,-0.87,1.57])
            if err<0.005: ys.append(y)
        print(f"   x={x} z=0.50 y in [{min(ys):.3f},{max(ys):.3f}]" if ys else f"   x={x} none")

# compare new vis_cone against committed version
import numpy as np, subprocess, importlib.util, sys
src = subprocess.check_output(["git","show","HEAD:approach.py"]).decode()
open("/tmp/old_appr.py","w").write(src)
sys.path.insert(0,"/tmp"); import old_appr, approach
rng_ = np.random.default_rng(0)
X, Y = np.meshgrid(np.arange(0.24, 2.3, 0.05), np.arange(-2.3, 2.3, 0.05), indexing="ij")
bad = 0
for t in range(30):
    tx, ty = rng_.uniform(-2.3, 2.3), rng_.uniform(-2.3, 2.3)
    occl = [(rng_.uniform(-2, 2), rng_.uniform(-2, 2), rng_.uniform(0.03, 0.3)) for _ in range(4)]
    for opt in (False, True):
        for rng in (2.0, 4.0):
            bc = rng_.uniform(-3, 3) + 2*np.pi/80*np.arange(80)
            a = old_appr.vis_cone(X, Y, tx, ty, 0.2, rng, occl, bc=bc, opt=opt)
            b = approach.vis_cone(X, Y, tx, ty, 0.2, rng, occl, bc=bc, opt=opt)
            bad += int((a != b).sum())
print("mismatches", bad)

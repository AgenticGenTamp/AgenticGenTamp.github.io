import numpy as np, sys
sys.path.insert(0,'.')
from approach import GeneratedApproach, R_hook
from kin import make_model, Kin
env, model = make_model()
obs,_ = env.reset(seed=0)
q0 = obs[128:135]; b0 = obs[125:128]
print("q0", np.round(q0,3), "b0", np.round(b0,3))
kin = Kin(model)
for a in (0.3,0.5,0.7,0.9):
    R = R_hook(a); dv=R[:,2]
    site = np.array([0.912,0,0.322]) - 0.02*dv + [0,0,0.08]
    q = kin.ik_multi(np.array([1.42,-0.33,np.pi]), q0, site, R)
    if q is None: print(a,"none"); continue
    d = q - q0
    print(a, np.round(d,2), "max", round(np.max(np.abs(d)),2))

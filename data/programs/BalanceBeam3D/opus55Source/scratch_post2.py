import numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import GeneratedApproach, HOME, down_rot
env = KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
ap = GeneratedApproach(env.action_space, env.observation_space, {})
kin=ap.kin
np.set_printoptions(precision=3, suppress=True)
H=np.array(HOME); base=np.zeros(3)
R0=ap.posture_R if hasattr(ap,'posture_R') else None
_,Rh=kin.fk(base,ap.posture(0.12))
for r in (0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65):
  for z in (0.12,0.03):
    best=None
    rng=np.random.default_rng(0)
    for trial in range(30):
        q0=H+ (rng.normal(0,0.6,7) if trial else 0)
        q,pe,re=kin.ik(base,np.array([r,0,z]),Rh,q0,bias=H)
        if pe<1e-3 and re<1e-2:
            m=np.abs(q-H).max()
            if best is None or m<best[0]: best=(m,q)
    print(r,z,best)

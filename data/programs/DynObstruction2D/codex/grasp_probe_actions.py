import numpy as np
from env_client import make_env


def one(seed, actions, label):
    env = make_env()
    s, _ = env.reset(seed=seed)
    rob = s.get_objects(env.observation_space.get_type("kin_robot"))[0]
    blk = s.get_objects(env.observation_space.get_type("target_block"))[0]
    print("\n", label, "seed", seed)
    def status(i, r=None, term=False):
        print(i, "R", *(round(s.get(rob,k),3) for k in ("x","y","theta","arm_length","finger_gap")),
              "B", *(round(s.get(blk,k),3) for k in ("x","y","theta","held")), "rew", r, "term",term)
    status(0)
    for i,a in enumerate(actions,1):
        s,r,t,tr,_=env.step(np.asarray(a,dtype=np.float32))
        if i <= 5 or i%5==0 or s.get(blk,"held") or t: status(i,r,t)
        if t or tr: break
    env.close()


zero=[0,0,0,0,0]
for seed in range(4):
    one(seed, [[0,0,0,0,-.02]]*20, "close")
one(0, [[0,0,0,.1,0]]*5 + [[0,0,0,0,-.02]]*20, "extend close")
one(0, [[.05,-.05,0,0,0]]*3 + [[0,0,0,0,-.02]]*20, "diag close")

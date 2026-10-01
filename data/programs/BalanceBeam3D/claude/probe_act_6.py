import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
env=make_env()
# stall detection with tolerance
for j,d in enumerate(range(3,10)):
    for sgn in [1,-1]:
        obs,_=env.reset(seed=0); a=np.zeros(11,dtype=np.float32); a[d]=0.1*sgn
        q=[obs[19+j]]
        for i in range(200): obs,r,te,tu,_=env.step(a); q.append(obs[19+j])
        q=np.array(q); dq=np.abs(np.diff(q))
        stalled=np.where(dq<0.002)[0]
        st=stalled[0] if len(stalled) and stalled[0]>3 else None
        print(f"j{j+1}{'+' if sgn>0 else '-'}: final {q[-1]:.4f} stall_step {st} val_at_stall {q[st] if st else None}")
env.close()

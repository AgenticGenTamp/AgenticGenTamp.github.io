from env_client import make_env
import numpy as np

e=make_env();s,i=e.reset(seed=0)
rob=s.get_object_from_name('robot')
for k in range(6):
    if k:
        for j in range(10):
            a=np.zeros(11);a[4]=-.1;a[6]=.1;a[10]=1
            s,r,te,tr,info=e.step(a)
    print(k,[round(s.get(rob,'pos_arm_joint'+str(j)),3) for j in range(1,8)], e.render_state(state=s,label='pose_'+str(k)),flush=True)
e.close()

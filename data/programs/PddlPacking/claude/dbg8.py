import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach, Robot
import fk
env=make_env(); obs,info=env.reset(seed=2,options={"object_count":4})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for k in range(74):
    a=ap.get_action(obs); obs,r,t,tr,i=env.step(a)
R=Robot(obs)
p,Rw,_=R.tool(); print("tool",np.round(p,3))
# what does cart_step want?
tgt=np.array([-0.088,0.0,1.05])
for i in range(8):
    dq,conv=ap.cart_step(R,tgt,Rw,step_len=0.08)
    print(i,"conv",conv,"dq",None if dq is None else np.round(dq,3))
    if dq is None or conv: break
    prev=R.q.copy(); pb=R.base.copy()
    obs,r,t,tr,inf=env.step(ap.act_arm(dq)); R=Robot(obs)
    print("   moved" if not np.allclose(prev,R.q) else "   REJECTED", np.round(R.tool()[0],3))
    if np.allclose(prev,R.q):
        # try base move in y
        obs,r,t,tr,inf=env.step(np.float32([0,0.1,0,0,0,0,0,0,0,0,0])); R2=Robot(obs)
        print("   base y move:", "ok" if not np.allclose(R2.base,pb) else "REJ", np.round(R2.tool()[0],3))
        R=R2
        # try lifting tool
        dq2,_=ap.cart_step(R,np.array([p[0],p[1],p[2]+0.12]),Rw,step_len=0.08)
        if dq2 is not None:
            prev=R.q.copy(); obs,r,t,tr,inf=env.step(ap.act_arm(dq2)); R=Robot(obs)
            print("   lift:", "ok" if not np.allclose(prev,R.q) else "REJ", np.round(R.tool()[0],3))
        break
env.close()

from env_client import make_env
import numpy as np

env=make_env()
for change in ['zero','j1+','j1-','j2+','j2-','j3+','j3-','j4+','j4-','j5+','j5-','j6+','j6-','j7+','j7-','rot+','rot-']:
    s,_=env.reset(seed=779)
    initial=s.copy()
    for i in range(20):
        a=np.zeros(11,dtype=np.float32)
        if change=='zero':a[3:10]=np.clip(-s[3:10],-.4,.4)
        elif change.startswith('rot'):a[2]=.4 if change[-1]=='+' else -.4
        else:a[2+int(change[1])]=.4 if change[-1]=='+' else -.4
        if i>=8:a[:2]=np.clip(s[19:21]-s[:2],-.4,.4)
        s,r,done,trunc,_=env.step(a)
        if done:break
    print(change,done,i+1,s[:10].tolist(),flush=True)
env.close()

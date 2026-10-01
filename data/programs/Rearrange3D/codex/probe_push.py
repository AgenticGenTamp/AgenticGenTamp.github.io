from env_client import make_env
import numpy as np

env = make_env()
o, _ = env.reset(seed=0)
base_obj = o.copy()
print('init', np.round(o[[0,1,2,16,17,18,32,33,34,93,94,95,103]],3))

def run(name, action, n):
    global o
    for k in range(n):
        old=o.copy(); o,r,t,tr,_=env.step(np.array(action,np.float32))
        dp=o[[0,1,2,16,17,18,32,33,34]]-old[[0,1,2,16,17,18,32,33,34]]
        if np.max(np.abs(dp)) > .002 or k==n-1 or r != -1:
            print(name,k+1,'r',r,'obj',np.round(o[[0,1,2,16,17,18,32,33,34]],3),'dp',np.round(dp,3),'rob',np.round(o[[93,94,95,103]],3))

# sweep gripper/arm assembly sideways through box, close, retreat
run('open', [0,0,0,0,0,0,0,0,0,0,0], 3)
run('+y', [0,.1,0,0,0,0,0,0,0,0,0], 4)
run('close', [0,0,0,0,0,0,0,0,0,0,1], 8)
run('-y', [0,-.1,0,0,0,0,0,0,0,0,1], 5)
run('+x', [.1,0,0,0,0,0,0,0,0,0,1], 6)
env.close()

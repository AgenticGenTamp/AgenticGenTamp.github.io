import sys, numpy as np
from common import make_env, hook
from approach import GeneratedApproach, LIM
sd=int(sys.argv[1]); pen=float(sys.argv[2]); spd=float(sys.argv[3]); cnt=int(sys.argv[4]) if len(sys.argv)>4 else 1
env=make_env(); s,info=env.reset(seed=sd,options={'object_count':cnt})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
ap._parse(s); X=ap.hook['x']
def go(tx,ty,th=-np.pi/2,arm=0.4,gap=0.09,n=200):
    global s
    for _ in range(n):
        ap._parse(s); r=ap.robot
        d=np.array([tx-r['x'],ty-r['y'],th-r['theta'],arm-r['arm_joint'],gap-r['finger_gap']])
        if np.abs(d).max()<1e-3: return
        sc=max(1,np.max(np.abs(d)/(LIM*0.999)))
        s,*_=env.step((d/sc).astype(np.float32))
# finger tip y = base - 0.4 - 0.02 - 0.12 ; bar top = 0.05
by = 0.05 - pen + 0.54
go(X-0.25, 1.0); go(X-0.25, by+0.05); go(X-0.25, by, n=30)
ap._parse(s); print('start', round(ap.hook['x'],4), round(ap.robot['y'],3))
for t in range(40):
    s,*_=env.step(np.array([-spd,0,0,0,0],dtype=np.float32))
    ap._parse(s)
    if t%5==4: print(t, round(ap.hook['x'],4), round(ap.hook['y'],4), round(ap.hook['theta']+np.pi/2,4), round(ap.robot['x'],3), round(ap.robot['y'],3))

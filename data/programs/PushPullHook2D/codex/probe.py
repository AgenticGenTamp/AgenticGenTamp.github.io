from env_client import make_env
import numpy as np

def run(seed, actions):
    e=make_env(); o,_=e.reset(seed=seed)
    print('start', np.round(o[[0,1,2,4,6,9,10,11,20,21,29,30]],4))
    for name,a,n in actions:
        for i in range(n):
            old=o.copy(); o,r,t,tr,info=e.step(np.array(a,np.float32))
        print(name,n,'key',np.round(o[[0,1,2,4,6,9,10,11,20,21,29,30]],4),'delta',np.round((o-old)[[0,1,2,4,6,9,10,11,20,21]],4),r,t,tr,info)
    e.close()

run(0,[('dx',[.05,0,0,0,0],1),('dy',[0,.05,0,0,0],1),('rot',[0,0,.196,0,0],1),('arm',[0,0,0,.1,0],1),('vac',[0,0,0,0,1],1),('armout',[0,0,0,.1,1],10)])

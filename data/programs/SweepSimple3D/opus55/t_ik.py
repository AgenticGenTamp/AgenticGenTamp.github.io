import numpy as np, kin, time
rng=np.random.default_rng(0)
bp=(1.0,1.0,-np.pi/2); n=0; ok_n=0; t=time.time(); its=[]
for i in range(200):
    tgt=np.array([1.0+rng.uniform(-.3,.3),1.0-rng.uniform(0.3,0.75),rng.uniform(0.0,0.4)])
    q,ok,e=kin.ik(bp,kin.Q_HOME,tgt,'down',yaw=rng.uniform(-np.pi,np.pi))
    n+=1; ok_n+=ok
    if not ok and i<400: print('fail',np.round(tgt,2),round(e,4))
print(ok_n,'/',n,'time per ik ms',(time.time()-t)/n*1000)

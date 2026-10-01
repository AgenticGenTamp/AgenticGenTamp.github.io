from calib_util import *
env=make_env()
for K in [0.7,1,1.3,1.6,2]:
  for amp in [0.05,0.3,1.0]:
    obs,_=env.reset(seed=0); b,q0=rstate(obs)
    qt=q0+amp*np.array([1,1,-1,1,1,1,-1])
    hist=[]
    for k in range(120):
        b,q=rstate(obs); e=qt-q; hist.append(np.max(np.abs(e)))
        obs,_,_,_,_=env.step(act(dq=np.clip(K*e,-0.1,0.1)))
    h=np.array(hist)
    def first(th):
        i=np.where(h<th)[0]; return i[0] if len(i) else -1
    print(K,amp,'<0.01',first(0.01),'<0.003',first(0.003),'<0.001',first(0.001),'max last40',h[-40:].max().round(4))

from calib_util import *
env=make_env()
for K in [1,2,3,4,6,10]:
    obs,_=env.reset(seed=0); b,q0=rstate(obs)
    qt=q0+np.array([0.3,0.3,-0.3,0.3,0.2,0.3,-0.3])
    hist=[]
    for k in range(80):
        b,q=rstate(obs); e=qt-q; hist.append(np.max(np.abs(e)))
        obs,_,_,_,_=env.step(act(dq=np.clip(K*e,-0.1,0.1)))
    h=np.array(hist)
    def first(th):
        i=np.where(h<th)[0]; return i[0] if len(i) else -1
    print(K,'steps<0.01',first(0.01),'<0.003',first(0.003),'<0.001',first(0.001),'final',h[-5:].round(4), 'overshoot?')

import numpy as np
from env_client import make_env

def get(s,e,typ): return s.get_objects(e.observation_space.get_type(typ))[0]
def q(s,o,*fs): return tuple(round(float(s.get(o,f)),6) for f in fs)

def run(seed, vacuum_early=False):
    e=make_env(); s,_=e.reset(seed=seed); r=get(s,e,"crv_robot"); b=get(s,e,"target_block")
    bx=float(s.get(b,"x")); print("start",seed,q(s,r,"x","y","theta","arm_joint"),q(s,b,"x","y","theta","width","height"))
    k=0
    # align exactly in x using bounded deltas, then descend base
    while abs(float(s.get(r,"x"))-bx)>1e-5 and k<30:
        dx=np.clip(bx-float(s.get(r,"x")),-.05,.05)
        s,*_=e.step(np.array([dx,0,0,0,float(vacuum_early)],np.float32)); k+=1
    print("aligned",k,q(s,r,"x","y","arm_joint"))
    for i in range(16):
        old=q(s,r,"x","y","arm_joint","vacuum"); ob=q(s,b,"x","y","theta")
        s,re,te,tr,info=e.step(np.array([0,-.05,0,0,float(vacuum_early)],np.float32))
        print("down",i,old,"=>",q(s,r,"x","y","arm_joint","vacuum"),"block",ob,"=>",q(s,b,"x","y","theta"))
    for i in range(3):
        old=q(s,r,"x","y","arm_joint","vacuum"); ob=q(s,b,"x","y","theta")
        s,re,te,tr,info=e.step(np.array([0,0,0,.1,1],np.float32))
        print("extend",i,old,"=>",q(s,r,"x","y","arm_joint","vacuum"),"block",ob,"=>",q(s,b,"x","y","theta"))
    for i in range(5):
        ob=q(s,b,"x","y","theta"); old=q(s,r,"x","y","arm_joint","vacuum")
        s,re,te,tr,info=e.step(np.array([.03,.03,.1,-.05,1],np.float32))
        print("carry",i,old,"=>",q(s,r,"x","y","theta","arm_joint","vacuum"),"block",ob,"=>",q(s,b,"x","y","theta"))
    e.close()

if __name__=="__main__": run(0,True)

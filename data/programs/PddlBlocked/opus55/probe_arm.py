from env_client import make_env
import numpy as np
from kin import *
from envutil import Sim
env = make_env()
S=Sim(env,env.reset(seed=0)[0])
def tryjoints(label):
    for j in range(7):
        for s in [0.1,-0.1]:
            q0=S.q(); a=np.zeros(11); a[3+j]=s; S.step(a)
            moved=np.abs(S.q()-q0).max()>1e-6
            print(label,'j',j+1,s,'moved' if moved else 'BLOCKED')
            if moved: a[3+j]=-s; S.step(a)
tryjoints('start')
S.moveto(np.array([3.85,-0.04,0]),S.q())
tryjoints('at table')

from kin_util import *
import math
def go(obs,dx,dy,n=200):
    prev=R(obs); hist=[]
    for k in range(n):
        obs,*_=env.step(A(dx,dy)); cur=R(obs)
        hist.append(cur)
        if cur==prev: break
        prev=cur
    return obs,hist
for seed in [0,1,2]:
    obs,_=env.reset(seed=seed); print('seed',seed,R(obs))
    for d in [(0.05,0),(-0.05,0),(0,-0.05),(0,0.05)]:
        obs0=obs
        obs,h=go(obs,*d); print(d,'last few',h[-3:],len(h))

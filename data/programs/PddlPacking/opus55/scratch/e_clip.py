import sys; sys.path.insert(0,"scratch"); from e_z import *
o,_=env.reset(seed=0); s=rstate(o)
for v in [0.15,0.231,0.5]:
    a=np.zeros(11,np.float32); a[0]=v; a[4]=-v; o2=step(o,a); print(v,(rstate(o2)-s)[[0,4]].round(4)); o,_=env.reset(seed=0)

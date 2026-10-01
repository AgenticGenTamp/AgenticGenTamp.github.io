import sys; sys.path.insert(0,"scratch"); from e_z import *
for i,v in [(3,0.5),(1,0.8),(2,0.6),(6,0.5)]:
    o,_=env.reset(seed=0); s=rstate(o)
    a=np.zeros(11,np.float32); a[i]=v; o2=step(o,a); print(i,v,(rstate(o2)-s)[i].round(4))

import numpy as np, pickle, sys
import kin
data=pickle.load(open(sys.argv[1],'rb'))
A=[];b=[]
for base,q,c,_ in data:
    Rb=kin.rotz(base[2]); p0,R0=kin.fk_arm(q,tool=0.0)
    A.append(np.hstack([Rb, Rb@R0])); b.append(c-np.array([base[0],base[1],0])-Rb@p0)
A=np.vstack(A); b=np.concatenate(b)
x,res,rk,sv=np.linalg.lstsq(A,b,rcond=None)
print('n',len(data),'rank',rk,'sv',sv.round(3))
print('mount',x[:3].round(4),'t(bracelet frame)',x[3:].round(4))
print('resid rms', np.sqrt(np.mean((A@x-b)**2)).round(5), 'max', np.abs(A@x-b).max().round(4))

import sys; sys.path.insert(0,"scratch"); from e_q4 import *
q=np.pi/4; r2=np.sqrt(.5)
o,A,Bb=setupAB(); sweep(o,A,Bb,q,[0.08,0.084,0.0845,0.085,0.09])
o,A,Bb=setupAB(); sweep(o,A,Bb,q,[0.085,0.09,0.1,0.11,0.115,0.12,0.125],dirn=(r2,r2))
o,A,Bb=setupAB(); sweep(o,A,Bb,0.0,[0.09,0.095,0.098,0.0995,0.1,0.11],dirn=(r2,r2))

import numpy as np
from helper import E
e=E(0)
print(e.goto(0.0,-1.5), e.pos())
# from -1.5 test single steps of various size
import copy
for d in [0.1,0.2,0.3,0.35,0.4]:
    e2=E(0); e2.goto(0.0,-1.5)
    m,t=e2.move(0,-d)
    print("step",d,"moved",m,"pos",e2.pos())
# fine approach downward
e3=E(0); e3.goto(0.0,-1.5)
lo=-1.5
for d in [0.05]*20:
    m,t=e3.move(0,-0.02)
    if not m: break
print("min y at x=0:", e3.pos())

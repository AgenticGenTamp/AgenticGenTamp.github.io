from ik_tool import *
import numpy as np
q1=np.array([0.0029, 1.52412, 3.14236, -1.74571, 0.00598, 0.12824, 1.56783])
for t in np.linspace(0,1,11):
    q=q_init+(q1-q_init)*t; x,R=fk(q); print(round(t,1), x.round(3), R[:,2].round(2))

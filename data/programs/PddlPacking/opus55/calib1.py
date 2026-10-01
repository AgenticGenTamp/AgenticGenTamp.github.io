# Check initial state FK with various torso heights; can't observe tool, just print
from pr2fk import *
q0=[0.6772,-0.3431,1.2,-1.4669,1.2422,-1.9544,2.2225]
t,R=fk((-1,0,0),q0)
print(t); print(R.round(3))

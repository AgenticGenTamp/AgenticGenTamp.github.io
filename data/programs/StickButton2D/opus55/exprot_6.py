from exprot_4 import *
o=replay(seed,acts)
for i in range(8): o=step(o,dth=-math.pi/16)
o=goto(o,1.2,1.006); o=goto(o,1.2,1.15)
while rd(o)[0]['theta']<1.89-1e-6: o=step(o,dth=min(0.196,1.891-rd(o)[0]['theta']))
r,s,b=rd(o); print('start th',round(r['theta'],3),'dist',round(dist(s,2.113,1.801),4),[x[4] for x in b if x[0]=='button0'])
o=step(o,dth=0.196); r,s,b=rd(o); print('after +0.196 th',round(r['theta'],3),'dist',round(dist(s,2.113,1.801),4),'b0 g',[x[4] for x in b if x[0]=='button0'])

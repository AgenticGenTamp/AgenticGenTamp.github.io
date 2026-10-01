import sys; sys.path.insert(0,"scratch"); from h import *
o,info=env.reset(seed=0); s=rstate(o)
for x in np.arange(-0.6,0,0.02):
    o,r=goto(o,(x,0,0),s[3:10])
    if r: print("base rej at",x, rstate(o)[:3]); break
for y in [-0.3]:
  o,info=env.reset(seed=0)
  for x in np.arange(-0.6,0,0.02):
    o,r=goto(o,(x,-0.3,0),s[3:10])
    if r: print("base y-.3 rej at",x, rstate(o)[:3]); break

from probe_lib import *
o,_=env.reset(seed=0); print('start',rob(o))
o,*_=env.step(A(0.05,0.0)); print('dx.05',rob(o))
o,*_=env.step(A(0.0,0.03)); print('dy.03',rob(o))
o,*_=env.step(A(0,0,0.1)); print('dth.1',rob(o))
o,*_=env.step(A(0,0,0,0.05)); print('darm.05',rob(o))
for i in range(5): o,*_=env.step(A(0,0,0,0.1))
print('arm max',rob(o))
for i in range(5): o,*_=env.step(A(0,0,0,-0.1))
print('arm min',rob(o))
o,*_=env.step(A(0,0,0,0,0.7)); print('vac .7',rob(o))
o,*_=env.step(A(0,0,0,0,0.3)); print('vac .3',rob(o))
for i in range(20): o,*_=env.step(A(0,0,0.196))
print('theta wrap',rob(o))
# bounds
o,_=env.reset(seed=0)
for d,(ax,ay) in {'-x':(-.05,0),'+x':(.05,0)}.items():
    for i in range(60): o,*_=env.step(A(ax,ay))
    print(d,rob(o))
o,_=env.reset(seed=0)
for d,(ax,ay) in {'-y':(0,-.05),'+y':(0,.05)}.items():
    for i in range(60): o,*_=env.step(A(ax,ay))
    print(d,rob(o))

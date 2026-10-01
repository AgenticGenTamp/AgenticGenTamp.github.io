from kin_util import *
obs,_=env.reset(seed=0); print(R(obs))
for a in [A(0.05),A(0,0.05),A(0.03,-0.02),A(dt=0.196),A(da=0.1),A(da=-0.1),A(v=1),A(v=0.4),A(v=0.6),A(v=0)]:
    o2,r,t,tr,i=env.step(a); print(a, R(o2), r,t,tr); obs=o2
for k in range(40):
    obs,*_=env.step(A(dt=0.196))
    if k%4==0: print('th',R(obs)[2])
s=[]
for k in range(10): obs,*_=env.step(A(da=0.1)); s.append(R(obs)[3])
print('arm+',s); s=[]
for k in range(12): obs,*_=env.step(A(da=-0.1)); s.append(R(obs)[3])
print('arm-',s)

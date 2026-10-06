import sys, numpy as np, mujoco
exec(open('grasptest.py').read().split("run([ox,0,0.6],8")[0])
zp=float(sys.argv[4])
run([0,0,0.6],6,1.0)
run([0,0,0.5],3,1.0)
run([0,0,zp],4,1.0)
print('drag')
for k in range(1,5):
    run([0.012*k,0,zp],1,1.0)
run([0.05,0,zp],3,1.0)
run([0.05,0,0.55],3,1.0)

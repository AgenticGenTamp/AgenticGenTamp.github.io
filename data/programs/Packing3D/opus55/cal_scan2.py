from cal_trial import *
import sys, time, itertools
dz=float(sys.argv[1]); yaw=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
vals=[-0.03,-0.02,-0.01,0,0.01,0.02,0.03]
t0=time.time()
print('dz',dz,'yaw',yaw,'rows dy (top=+), cols dx', vals)
for dy in vals[::-1]:
    row=''
    for dx in vals:
        t=trial(0,'part0',dx,dy,dz,yaw=yaw)
        row+=('G' if t['ga']>0 else '.')+('' if t['reached'] else '*')+' '
    print(f'dy={dy:+.2f} {row}',flush=True)
print('time',time.time()-t0)

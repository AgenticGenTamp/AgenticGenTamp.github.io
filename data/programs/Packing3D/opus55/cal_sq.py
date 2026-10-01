from cal_trial import *
for dx,dy in [(0.017,0.017),(-0.017,-0.017),(0.013,0.013),(-0.013,0.013)]:
    t=trial(0,'part0',dx,dy,0.05); print(dx,dy,'ga',t['ga'],flush=True)

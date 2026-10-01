from cal_trial import *
vals=[-0.05,-0.03,-0.01,0.01,0.03,0.05]
print('rows dy top=+, cols dx',vals)
for dy in vals[::-1]:
    row=''
    for dx in vals:
        t=trial(0,'part1',dx,dy,0.05); row+=('G' if t['ga']>0 else '.')+('' if t['reached'] else '*')+' '
        if t['ga']>0: gg=t['gtf']
    print(f'dy={dy:+.2f} {row}',flush=True)

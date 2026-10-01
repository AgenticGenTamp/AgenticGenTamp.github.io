from cal_trial import *
import sys
seed=int(sys.argv[1]); pn=sys.argv[2]; cx=float(sys.argv[3]); cy=float(sys.argv[4])
offs=np.arange(-0.03,0.0301,0.005)
r1=''.join('G' if trial(seed,pn,cx+o,cy,0.05)['ga']>0 else '.' for o in offs); print('x-scan',r1,flush=True)
r2=''.join('G' if trial(seed,pn,cx,cy+o,0.05)['ga']>0 else '.' for o in offs); print('y-scan',r2,flush=True)
print('offs',np.round(offs,3))

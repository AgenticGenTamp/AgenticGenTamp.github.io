import numpy as np, pickle, glob
from approach import fk_arm
res = []
for f in glob.glob('opt2_res_free_h_*.pkl'): res += pickle.load(open(f, 'rb'))
t, x = min(res, key=lambda a: a[0])
w = lambda q: np.round((q + np.pi) % (2*np.pi) - np.pi, 3).tolist()
print(t, 'q1', w(x[:7]), '\nqa(0.61)', w(x[7:14]), '\nqb(0.69)', w(x[19:26]))
print('R2 x-axis', fk_arm(x[7:14])[:3, 0].round(3), 'z', fk_arm(x[7:14])[:3, 2].round(3))

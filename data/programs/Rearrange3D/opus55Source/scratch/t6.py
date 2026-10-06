import sys; sys.path.insert(0,'/sandbox/scratch'); from h import *
for s in [36,49,53,60]:
    h=H(s); out=[]
    for k in range(10): h.step(50); out.append(np.round(h.drink()[:2],3).tolist())
    print(s, out)

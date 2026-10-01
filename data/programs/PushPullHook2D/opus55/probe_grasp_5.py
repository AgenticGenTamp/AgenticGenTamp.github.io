from probe_grasp_2 import trial
import sys
tot=ok=0; odd=[]
for seed in list(range(int(sys.argv[1]),int(sys.argv[2]))):
    for a in [0.3,0.6,0.9,1.15]:
        for side in [1,-1]:
            r=trial(seed,a,side)
            if not r.startswith('b='): continue
            tot+=1
            if 'G' in r: ok+=1
            else: odd.append((seed,a,side,r))
            if 'G' in r and 'x' in r: odd.append((seed,a,side,r))
print('contact trials',tot,'grasped',ok); print(odd)

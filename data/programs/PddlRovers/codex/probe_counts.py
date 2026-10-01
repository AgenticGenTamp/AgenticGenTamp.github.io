from collections import Counter
from env_client import make_env

c = Counter(); patterns=Counter(); bad=[]
for seed in range(50):
    e = make_env(); s, info = e.reset(seed=seed)
    c[info['object_count']] += 1
    typ=e.observation_space.get_type('sample')
    pat=[]
    for soil in (0,1):
        xs=[s.get(o,'x') for o in s.get_objects(typ) if int(s.get(o,'is_soil')>.5)==soil]
        sides=('L' if any(x<-.3 for x in xs) else '')+('R' if any(x>.3 for x in xs) else '')+('C' if any(abs(x)<=.3 for x in xs) else '')
        pat.append(sides)
    patterns[tuple(pat)]+=1
    # opposite-kind, one task each is side-feasible iff (R stone,L soil) or vice versa
    if not ((('R' in pat[0]) and ('L' in pat[1])) or (('R' in pat[1]) and ('L' in pat[0]))): bad.append((seed,pat))
    e.close()
print(c); print(patterns); print('bad',bad)

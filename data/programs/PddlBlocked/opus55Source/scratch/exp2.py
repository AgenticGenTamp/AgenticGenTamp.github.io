exec(open('scratch/exp1.py').read().replace('for r in sorted(res): print(r)','pass').replace('res.append((','res.append((q,'))
res.sort(key=lambda r:r[1])
print("obst", e.obstacles, "mov", e.movables, "walls", e.walls)
for r in res[:12]:
    print(r[1:], diag(r[0])[:8])

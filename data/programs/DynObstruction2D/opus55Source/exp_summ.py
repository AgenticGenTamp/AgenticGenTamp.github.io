import sys, re
for f in sys.argv[1:]:
    tot = ok25 = ok30 = boom = 0
    for l in open(f):
        m = re.search(r"gap ([-\d.]+)->([-\d.e+]+) th=([-\d.e+]+) y=([-\d.e+]+)", l)
        if not m: continue
        tot += 1
        g1, th, y = float(m.group(2)), float(m.group(3)), float(m.group(4))
        if abs(y) > 2: boom += 1
        ww = re.search(r"w=([\d.]+) h=([\d.]+)", l); w, h = float(ww.group(1)), float(ww.group(2))
        import math
        t = th % (math.pi / 2); t = min(t, math.pi / 2 - t)
        hh = h / 2 if abs(math.cos(th)) > 0.7 else w / 2
        flat = t < 0.05 and abs(y - (0.1 + hh)) < 0.012
        ok25 += flat and g1 >= 0.25; ok30 += flat and g1 >= 0.30
    print(f, "n", tot, "ok>=.25", ok25, "ok>=.30", ok30, "explode", boom)

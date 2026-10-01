s = open('opt2_validate.py').read()
s = s.replace("q, e = solve(np.array([d2, 0, zc]), R2, [qs]); qs = q", "sd = q2 + (zc - 0.61)/0.08*(x[19:26] - q2) if len(x) > 19 else qs\n        q, e = solve(np.array([d2, 0, zc]), R2, [sd]); qs = q")
open('opt2_validate.py','w').write(s)

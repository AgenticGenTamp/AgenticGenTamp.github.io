s = open('opt2_validate.py').read()
s = s.replace("""        q, e = solve(np.array([d2, 0, zc]), R2, [q2])""", """        q, e = solve(np.array([d2, 0, zc]), R2, [qs]); qs = q
        if abs(zc - 0.65) < 1e-6: qn = q""")
s = s.replace("worst = t; ok = True", "worst = 0; ok = True; qs = q2")
s = s.replace("out.update(zc_ok=ok,", "out.update(qn=qn, dqn=np.abs(qn - q1).max(), zc_ok=ok,")
s = s.replace("'\\n  q2', x[7:14].round(3).tolist())", "'\\n  q2(0.65)', v['qn'].round(3).tolist())")
s = s.replace("np.degrees(x[17])), v)", "np.degrees(x[17])), {k: (np.round(w, 4) if k != 'qn' else '') for k, w in v.items()})")
open('opt2_validate.py','w').write(s)

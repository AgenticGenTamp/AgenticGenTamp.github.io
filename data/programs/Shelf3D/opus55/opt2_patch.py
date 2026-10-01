s = open('opt2_search.py').read()
s = s.replace("def unpack(x): return x[:7], x[7:14], x[14], x[15], x[16], x[17], x[18]",
 "def unpack(x): return x[:7], x[7:14], x[14], x[15], x[16], x[17], x[18], x[19:26]")
s = s.replace("q1, q2, d1, p1, d2, p2, t = unpack(x)", "q1, q2, d1, p1, d2, p2, t, q3 = unpack(x)")
s = s.replace("""        if place_x_horiz: e.append(T2[2, 0])
        return np.array(e)""", """        if place_x_horiz: e.append(T2[2, 0])
        T3 = fk_arm(q3); c3 = cube(T3)
        e += [c3[0]-d2, c3[1], c3[2]-ZB] + list((T3[:3, :3] - T2[:3, :3]).ravel()[[2, 5, 0]])
        return np.array(e)""")
s = s.replace("return np.concatenate([t - dq, t + dq, [T1[0, 2], T2[0, 2]], cab_margins(q2, d2, zc)])",
 "dq3 = q3 - q1\n        return np.concatenate([t - dq, t + dq, t - dq3, t + dq3, [T1[0, 2], T2[0, 2]], cab_margins(q2, d2, zc), cab_margins(q3, d2, ZB)])")
s = s.replace("def run(zc=0.65,", "ZB = 0.69\ndef run(zc=0.61,")
s = s.replace("lb = np.r_[-LIM, -LIM, 0.4, np.radians(45), 0.6, np.radians(-10), 0]", "lb = np.r_[-LIM, -LIM, 0.4, np.radians(45), 0.6, np.radians(-10), 0, -LIM]")
s = s.replace("ub = np.r_[LIM, LIM, 0.8, np.radians(90), 0.95, np.radians(30), 5]", "ub = np.r_[LIM, LIM, 0.8, np.radians(90), 0.95, np.radians(30), 5, LIM]")
s = s.replace("""        if e1 > 5e-3 or e2 > 5e-3: continue
        x0 = np.r_[q1, q2, d1, p1, d2, p2, 1.0]""", """        q3, e3 = solve(np.array([d2, 0, ZB]), fk_arm(q2)[:3, :3], [q2])
        if e1 > 5e-3 or e2 > 5e-3 or e3 > 5e-3: continue
        x0 = np.r_[q1, q2, d1, p1, d2, p2, 1.0, q3]""")
open('opt2_search.py','w').write(s)

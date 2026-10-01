s = open('opt2_search.py').read()
a = s.index("        d1, p1, d2, p2 = rng"); b = s.index("        x0[18] = np.abs(q2 - q1).max()\n")
s = s[:a] + "        x0 = X0[k]\n" + s[b + len("        x0[18] = np.abs(q2 - q1).max()\n"):]
s = s.replace("    for k in range(n):", "    X0 = pickle.load(open('opt2_x0_%s.pkl' % CASE, 'rb'))\n    for k in range(len(X0)):")
s = s.replace("import numpy as np, sys, warnings", "import numpy as np, sys, warnings, pickle\nCASE = sys.argv[1] + '_' + sys.argv[2]")
s = s.replace("'opt2_res_%s_%s_%s.pkl'", "'opt2_res_%s_%s_ws%s.pkl'")
open('opt2_search_ws.py','w').write(s)

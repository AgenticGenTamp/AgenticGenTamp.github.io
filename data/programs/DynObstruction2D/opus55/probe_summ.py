import sys,ast,glob
rows=[ast.literal_eval(l) for f in glob.glob(sys.argv[1]) for l in open(f) if l.startswith('{')]
rows.sort(key=lambda r:r['e'])
ok=0
for r in rows:
    good = r['final_gap']>0.55 and abs(r['th'])<0.1 and r['maxth']<0.3
    ok+=good
    print('OK ' if good else 'BAD', r['seed'],r['side'],'e',r['e'],'h',r['h'],'w',r['w'],r['m'],'gap',r['final_gap'],'th',r['th'],'maxth',r['maxth'])
print(ok,len(rows))

import sys
exec(open('dbg3.py').read().split("w = ap.world; w.update(s)")[0])
w = ap.world; w.update(s)
held = ap._held()
print("held", held)
print(ap._column_status())
for qpre, qp in ap._place_configs(held, int(sys.argv[4]), int(sys.argv[5])) if held else []:
    ck_tight = Checker(w, held=held, margin=0.003)
    print([round(v,3) for v in qpre], [round(v,3) for v in qp], ck_tight.collides(qp), ck_tight.collides(qpre), ck_tight.path_ok(densify(qpre,qp)))

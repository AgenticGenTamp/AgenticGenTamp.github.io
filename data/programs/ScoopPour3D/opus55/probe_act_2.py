exec(open('probe_act_1.py').read().split('run([A(a0=-0.1)]')[0])
run([A(a2=0.1)]*10+[A(a0=-0.1)]+[z]*2+[A(a1=-0.1)]+[z]*2, "rot 1.0 then -x then -y")
run([A(a0=-0.05)]+[z]+[A(a0=-0.02)]+[z]+[A(a0=-0.01)]+[z], "scale")
run([A(a0=0.1)]*12+[z]*2+[A(a0=-0.1)]+[z]*2, "+x into island, then -x")

exec(open('probe_act_1.py').read().split('run([A(a0=-0.1)]')[0])
for j in [3,4,6,9]:
    run([A(**{f'a{j}':0.1})]+[z]*3, f"joint idx {j} once", idx=(j,))
run([A(a4=0.1)]*6+[z]*3, "joint1 (idx3) x6", idx=(3,4,5))
run([A(a6=-0.1)]*5+[z]*20, "idx6 -0.1 x5 hold 20 (gravity?)", idx=(4,6,8))
run([A(a10=1.0)]*8+[A(a10=0.0)]*8+[A(a10=0.5)]*4, "gripper", idx=(10,))

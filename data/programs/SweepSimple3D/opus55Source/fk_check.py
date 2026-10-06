from ik_tool import *
from approach import fk_pinch, arm_q
for q in [q_init, arm_q(0.05), arm_q(0.05)+0.1]:
    print(fk(q)[0].round(5), fk_pinch(q).round(5))

from probe_push_lib import *
p=P(135); p.goto(y=2.0); p.goto(gap=0.5); print(p.r()); p.goto(arm=0.6); print(p.r()); p.goto(arm=0.0,gap=0.0); print(p.r())

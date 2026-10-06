import numpy as np
from approach import Sim, Planner, CARRY_ARM
s=Sim(1); P=Planner(s)
s.set_state(s.get_state())
xs=np.arange(-1.2,1.21,0.05); ys=np.arange(-1.3,1.31,0.05)
for yaw in (0.0, np.pi/4, np.pi/2):
    print('yaw',round(yaw,2))
    for y in ys[::-1]:
        row=''
        for x in xs:
            q=np.concatenate([[x,y,yaw],CARRY_ARM])
            row+= '#' if P.table_collision(q) else '.'
        print(f'{y:5.2f} {row}')

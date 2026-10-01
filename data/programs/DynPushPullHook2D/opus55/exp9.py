from grasp import *
s=S(2)
print('held',grasp(s),s.rob(),s.pose('hook'))
s.goto(1.0,0.8,0.0); print('pos',s.rob(),s.pose('hook'))
for i in range(30): s.step([0,-0.05,0,0,0])
print('down',s.rob(),s.pose('hook'))
for i in range(40): s.step([0.05,0,0,0,0])
print('right',s.rob(),s.pose('hook'), s.g('hook','held'))

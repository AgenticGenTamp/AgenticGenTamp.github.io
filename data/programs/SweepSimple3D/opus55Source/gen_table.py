from ik_null import *
zs = [round(0.30+0.01*i,2) for i in range(21)]
print('HOLD_R = 0.45')
print('ARM_TABLE_Z =', zs)
print('ARM_TABLE_Q = [')
for z in zs:
    q,ep,er=ikn(np.array([0.45,0,z]),Rd,q_init,q_init)
    assert ep<1e-3 and er<1e-3
    print('    %s,'%np.round(q,5).tolist())
print(']')

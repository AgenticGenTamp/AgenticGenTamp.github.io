import numpy as np
A = dict(PLACE_ZOFF=0.04, PICK_PITCH=np.radians(82.1), PICK_D=0.554, PLACE_PITCH=np.radians(6.76), PLACE_D=0.95,
         Q_PICK0=np.array([0.205, 1.577, 2.888, -1.704, -0.692, 0.354, 2.471]),
         Q_PLACE_LO=np.array([0.351, 1.145, 2.530, -1.563, -0.003, 0.921, 2.127]),
         Q_PLACE_HI=np.array([0.442, 1.012, 2.342, -1.561, -0.168, 0.778, 2.338]))
B = dict(PICK_PITCH=np.radians(75.1), PICK_D=0.732, PLACE_PITCH=np.radians(15.6), PLACE_D=0.95,
         Q_PICK0=np.array([0.542, 2.018, -1.314, 1.198, -1.004, -1.015, 0.690]),
         Q_PLACE_LO=np.array([0.601, 1.680, -1.745, 1.526, -0.331, -0.999, 0.377]),
         Q_PLACE_HI=np.array([0.528, 1.680, -2.087, 1.495, -0.444, -1.161, 0.709]))
W = dict(PICK_PITCH=np.radians(77.5), PICK_D=0.770, PLACE_PITCH=np.radians(9.7), PLACE_D=1.15,
         Q_PICK0=np.array([0.010,1.645,3.123,-0.992,0.056,-0.286,1.533]),
         Q_PLACE_LO=np.array([0.002,1.469,3.138,-0.358,0.011,0.086,1.595]),
         Q_PLACE_HI=np.array([0.008,1.345,3.092,-0.328,0.158,-0.069,1.492]))
R = dict(PICK_PITCH=np.radians(73.6), PICK_D=0.780, PLACE_PITCH=np.radians(7.5), PLACE_D=1.15,
         Q_PICK0=np.array([0.083,1.654,2.974,-1.039,0.698,-0.208,1.024]),
         Q_PLACE_LO=np.array([-0.056,1.480,3.494,-0.383,0.152,0.157,1.243]),
         Q_PLACE_HI=np.array([-0.001,1.342,3.157,-0.372,0.345,0.013,1.384]))
T = dict(HOVER=0.03, APPROACH_TOL=0.12, GRIP_STEPS=2)

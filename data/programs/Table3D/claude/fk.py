import numpy as np

def dh(alpha, a, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0, a],
        [st*ca, ct*ca, -sa, -d*sa],
        [st*sa, ct*sa, ca, d*ca],
        [0,0,0,1.0]])

# Kinova Gen3 7DoF modified DH (Craig): alpha_{i-1}, a_{i-1}, d_i, theta_i
PI = np.pi
DH = [
    (PI,   0, -0.2848, 0.0),
    (PI/2, 0, -0.0118, PI),
    (PI/2, 0, -0.4208, PI),
    (PI/2, 0, -0.0128, PI),
    (PI/2, 0, -0.3143, PI),
    (PI/2, 0,  0.0,    PI),
    (PI/2, 0, -0.1674, PI),
]
TOOL = dh(PI, 0, -0.0615, 0.0)  # interface frame

def fk(q, tool_len=0.0):
    T = np.eye(4)
    for i in range(7):
        alpha, a, d, th = DH[i]
        T = T @ dh(alpha, a, d, th + q[i])
    T = T @ TOOL
    if tool_len:
        T = T @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,tool_len],[0,0,0,1.0]])
    return T

if __name__ == "__main__":
    print(np.round(fk(np.zeros(7)),4))
    home = np.array([0,-0.35,-3.1416,-2.5,0,-0.87,1.5708])
    print(np.round(fk(home),4))

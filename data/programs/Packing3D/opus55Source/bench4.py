import time, numpy as np
def tm(f,n=500):
    t=time.time()
    for _ in range(n): f()
    return round((time.time()-t)/n*1000,4)
A=np.random.rand(6,7); I=np.eye(6); b=np.ones(6)
print("solve", tm(lambda: A.T@np.linalg.solve(A@A.T+1e-3*I, b)))
print("matmul", tm(lambda: A@A.T))
print("solveonly", tm(lambda: np.linalg.solve(I*2, b)))
try:
    from threadpoolctl import threadpool_limits, threadpool_info
    print([ (i['internal_api'], i['num_threads']) for i in threadpool_info()])
    with threadpool_limits(1):
        print("solve1", tm(lambda: A.T@np.linalg.solve(A@A.T+1e-3*I, b)))
except Exception as e: print("no tpc", e)
from blas1 import limit_blas_threads
print("limited", limit_blas_threads(1))
print("solve after", tm(lambda: A.T@np.linalg.solve(A@A.T+1e-3*I, b)))

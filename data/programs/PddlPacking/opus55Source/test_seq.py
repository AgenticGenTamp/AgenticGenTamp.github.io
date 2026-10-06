import sys
import test_approach as T
k=int(sys.argv[1])
for s in [int(x) for x in sys.argv[2].split(',')]:
    print(s, T.run(s,k), flush=True)

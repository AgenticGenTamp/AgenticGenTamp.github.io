import sys, approach
approach.GeneratedApproach.DEBUG=True
import test_approach as T
print(T.run(int(sys.argv[1]), count=(int(sys.argv[2]) if len(sys.argv)>2 else None)))

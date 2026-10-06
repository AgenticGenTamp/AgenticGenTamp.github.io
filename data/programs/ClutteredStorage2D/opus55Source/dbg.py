import sys, time
from test_approach import run
import approach
n, s = int(sys.argv[1]), int(sys.argv[2])
import cProfile, pstats
cProfile.run("print(run(s, n))", "prof.out")
p = pstats.Stats("prof.out"); p.sort_stats("cumulative").print_stats(15)

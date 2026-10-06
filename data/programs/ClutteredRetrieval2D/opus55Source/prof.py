import cProfile, pstats, sys
from test_approach import run
cProfile.run(f"print(run({sys.argv[1]},{sys.argv[2]}))", "/tmp/prof.out")
p=pstats.Stats("/tmp/prof.out"); p.sort_stats("cumulative").print_stats(30)

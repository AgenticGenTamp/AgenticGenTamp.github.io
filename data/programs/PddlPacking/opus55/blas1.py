"""Limit OpenBLAS threads to 1 (tiny matrices; threading only adds overhead)."""
import os


def limit_blas_threads():
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    try:
        import ctypes
        libs = set()
        with open("/proc/self/maps") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 6 and ("openblas" in p[-1].lower() or "libblas" in p[-1].lower()):
                    libs.add(p[-1])
        for path in libs:
            try:
                lib = ctypes.CDLL(path)
            except OSError:
                continue
            for fn in ("openblas_set_num_threads", "openblas_set_num_threads64_",
                       "scipy_openblas_set_num_threads", "scipy_openblas_set_num_threads64_"):
                f = getattr(lib, fn, None)
                if f is not None:
                    try:
                        f(1)
                    except Exception:
                        pass
    except Exception:
        pass

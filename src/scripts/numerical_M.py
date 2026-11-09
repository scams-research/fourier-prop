import timeit
import numpy as np
import scipp as sc

from fourierror import numerical

import paths

N = 100
 
rng = np.random.default_rng(42)
 
M = 2 ** np.arange(start=5, stop=13, step=1, dtype=int)

def timed(fn, target_repeats=7):
    """Minimum per-call time, with `number` chosen automatically."""
    timer = timeit.Timer(fn)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=target_repeats, number=number)) / number

X = sc.linspace(dim="x", start=0, stop=2 * np.pi, num=N, unit="rad")
Y = sc.sin(2 * X) + 2 * sc.cos(10 * X) + 0.6 * sc.sin(20 * X)
Y += sc.array(dims=["x"], values=rng.standard_normal(N) * 0.1 * Y.values)
Y.variances = sc.abs(Y * 0.1)
Y.unit = sc.Unit("s")
data = sc.DataArray(data=Y, coords={"x": X})
 
mv_norm = numerical._generate_mv_norm(data)
dft_matrix = np.fft.fft(np.eye(data.values.size))
 
times = np.zeros((M.size, 2))
sampling = np.zeros(M.size)
 
for i, m in enumerate(M):
    m = int(m)
    samples = mv_norm.rvs(m).T
 
    times[i, 0] = timed(lambda: (2 / data.values.size) * (dft_matrix @ samples))
    times[i, 1] = timed(lambda: (2 / data.values.size) * np.fft.fft(samples, axis=0))
 
    sampling[i] = timed(lambda: mv_norm.rvs(m), target_repeats=3)
 
    total_dft = sampling[i] + times[i, 0]
    saving = (times[i, 0] - times[i, 1]) / total_dft * 100
 
    print(
        f"M={m:6d}  dft {times[i, 0] * 1e6:9.1f} us   "
        f"fft {times[i, 1] * 1e6:8.1f} us   "
        f"ratio {times[i, 0] / times[i, 1]:5.2f}   "
        f"rvs {sampling[i] * 1e6:9.1f} us   "
        f"saving {saving:4.0f}% of total"
    )
    print(times[i])

np.savez(paths.data / "numerical_M.npz", M=M, times=times)

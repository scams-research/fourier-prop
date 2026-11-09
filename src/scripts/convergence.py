import numpy as np
import scipp as sc
import matplotlib.pyplot as plt

from fourierror import numerical, analytical

import _fig_params as fp
import paths

# The same test signal as compare.py: 50 points with 30 % relative uncertainty.
X = sc.linspace(dim='x', start=0, stop=2*np.pi, num=50, unit='rad', endpoint=False)
Y = sc.sin(2 * X) + 2 * sc.cos(10 * X) + 0.6 * sc.sin(20 * X)
Y.variances = (0.3 * Y.values) ** 2
data = sc.DataArray(data=Y, coords={'x': X})

M = 2 ** np.arange(4, 21, 2)
REPEATS = 10
rng = np.random.default_rng(42)

exact = analytical.dft(data, 'x')


def covariance(da):
    """Covariance matrix of a transform output (full or diagonal)."""
    if hasattr(da, 'covariance'):
        return da.covariance.values
    return np.diag(da.variances)


def rel_error(approx, ref):
    """Relative Frobenius-norm error of a covariance matrix."""
    return np.linalg.norm(approx - ref) / np.linalg.norm(ref)


errors = np.zeros((2, M.size, REPEATS))
for i, m in enumerate(M):
    for j in range(REPEATS):
        F = numerical.dft(data, 'x', int(m), random_state=rng)
        for k, part in enumerate(['real', 'imag']):
            errors[k, i, j] = rel_error(covariance(F[part]), covariance(exact[part]))

fig, ax = plt.subplots(figsize=(fp.COLUMN_WIDTH, 2))
labels = [r'$\operatorname{Re}[\mathcal{F}(\omega)]$', r'$\operatorname{Im}[\mathcal{F}(\omega)]$']
for k in range(2):
    med = np.median(errors[k], axis=1)
    lo, hi = np.percentile(errors[k], [2.5, 97.5], axis=1)
    ax.plot(M, med, color=fp.colors[k + 1], marker='o', label=labels[k])
    ax.fill_between(M, lo, hi, color=fp.colors[k + 1], alpha=0.25, lw=0)

# Reference line through the first point, showing the expected M^(-1/2) decay.
ref = np.median(errors[0, 0]) * np.sqrt(M[0] / M)
ax.plot(M, ref * 2, ls='--', color='k', alpha=0.5, lw=0.5)
ax.text(M[-1] * 1.15, ref[-1] * 2, r'$\mathcal{O}(M^{-1/2})$', ha='left', va='center')

ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xticks(M[::2])
ax.set_xlabel('$M$')
ax.set_ylabel(r'$\|\Sigma_{\mathrm{num}} - \Sigma_{\mathrm{ana}}\|_F \,/\, \|\Sigma_{\mathrm{ana}}\|_F$')
ax.legend()

plt.savefig(paths.figures / "convergence.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close()

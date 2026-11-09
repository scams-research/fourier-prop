import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipp as sc
from scipy.special import spherical_jn

from fourierror import analytical
from fourierror.data_array import CovDataArray

import _fig_params as fp
import paths

data = paths.data / 'tip4p_hold.data'
traj = paths.data / 'tip4p_hold.atom'

u = mda.Universe(data, traj, format='LAMMPSDUMP', time_step=1.0, time_unit='ps')


def truncate_universe(u, start=0, stop=20):
    """Return a new Universe containing only the first n_frames."""
    ag = u.atoms
    coords, dims = [], []
    for ts in u.trajectory[start:stop]:
        coords.append(ag.positions.copy())
        dims.append(ts.dimensions.copy() if ts.dimensions is not None else None)

    new_u = mda.Merge(ag)                      # copies topology
    new_u.load_new(np.asarray(coords),         # shape (frames, atoms, 3)
                   format=MemoryReader,
                   dimensions=None if dims[0] is None else np.asarray(dims),
                   dt=u.trajectory.dt)
    return new_u

n = 50
n_molecules = 16

time = np.arange(n)
L_MAX = 2
# Legendre polynomials P_l(x) for l = 0, 1, 2.
LEGENDRE = [lambda x: np.ones_like(x), lambda x: x, lambda x: (3 * x ** 2 - 1) / 2]

# c[l, rep, t] = <P_l(u(0) . u(t))>, the rotational correlation function of
# order l for each sub-trajectory.
c = np.zeros((L_MAX + 1, len(u.trajectory) // n, time.shape[0]), dtype=float)

for rep in tqdm(np.arange(c.shape[1])):
    u_short = truncate_universe(u, start=0+(n*rep), stop=0+(n*rep)+n)

    vectors = np.zeros((len(u_short.trajectory), n_molecules, 3))

    for i, ts in enumerate(u_short.trajectory):
        hydrogen = u_short.select_atoms('type 2')[:n_molecules*2]
        vectors[i] = hydrogen[1::2].positions - hydrogen[0::2].positions

    unit = vectors / np.linalg.norm(vectors, axis=-1)[:, :, np.newaxis]


    for t in range(vectors.shape[0]):
        dots = np.einsum("fvd,fvd->fv", unit[: vectors.shape[0] - t], unit[t:])
        for l in range(L_MAX + 1):
            c[l, rep, t] = np.mean(LEGENDRE[l](dots))

# The first-order function, C_1(t), is shown in panels (a) and (b).
c_mean = c[1].mean(0)
c_cov = np.cov(c[1].T)

hh_distance = 1.5139
q = np.array([1.0])

# Rotational intermediate scattering function,
#     F(q, t) = sum_l (2l + 1) j_l(q d)^2 C_l(t),
# built for each sub-trajectory so that its full covariance, including the
# correlation between the l terms, is carried into the transform.
ells = np.arange(L_MAX + 1)
weights = (2 * ells + 1) * spherical_jn(ells, q[0] * hh_distance) ** 2
isf = np.einsum('l,lrt->rt', weights, c)
data = CovDataArray(
    data=sc.array(dims=['time'], values=isf.mean(0)),
    coords={'time': sc.array(dims=['time'], values=time, unit='ps')},
    covariance=np.cov(isf.T),
)

def plot_spectrum(ds, ax, label, coord, **kw):
    """Plot a spectrum, sorted by frequency, with a 95 % confidence band (1.96 sigma).

    The omega = 0 point is omitted; it holds the elastic (l = 0) contribution.
    """
    w = ds.coords[coord].values
    order = np.argsort(w)
    keep = order[~np.isclose(w[order], 0)]
    v = ds.values[keep]
    e = np.sqrt(ds.variances[keep])
    ax.plot(w[keep], v, label=label, **kw)
    ax.fill_between(w[keep], v - 1.96 * e, v + 1.96 * e, alpha=0.25, lw=0, **kw)

fig = plt.figure(figsize=(fp.COLUMN_WIDTH, 3.4))
gs = plt.GridSpec(2, 2, figure=fig, wspace=0.6, hspace=0.5)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0]) 
ax4 = fig.add_subplot(gs[1, 1])

c_std = np.sqrt(np.diag(c_cov))
ax1.plot(time, c_mean, color=fp.colors[0])
ax1.fill_between(time, c_mean - 1.96 * c_std, c_mean + 1.96 * c_std, alpha=0.25, lw=0, color=fp.colors[0])
ax1.set_xlabel(f"$t$ / {data.coords['time'].unit}")
ax1.set_ylabel(r'$C_{1}(t)$')
ax1.set_yticks([0, 0.5, 1])

fp.matrix_map(fig, ax2, time, time, c_cov, r'$\Sigma[C_{1}(t)]$')
ax2.set_xticks([0, 20, 40])
ax2.set_yticks([0, 20, 40])
ax2.set_xlabel(f"$t$ / {data.coords['time'].unit}")
ax2.set_ylabel(f"$t$ / {data.coords['time'].unit}")

ll = analytical.dft(data, 'time')

plot_spectrum(ll['real'], ax3, rf'$q={q[0]}$' + ' Å$^{-1}$', 'omega', color=fp.colors[1])
ax3.set_xlabel(r"$\omega$ / rad ps$^{-1}$")
ax3.set_xticks([-2, 0, 2])
ax3.set_ylabel(fr'$S(\omega)$')

w = ll['real'].coords['omega'].values
order = np.argsort(w)
keep = order[~np.isclose(w[order], 0)]

cov_s = ll['real'].covariance.values[keep][:, keep]
fp.matrix_map(fig, ax4, w[keep], w[keep], cov_s, r'$\Sigma[S(\omega)]$')
ax4.set_xticks([-2, 0, 2])
ax4.set_yticks([-2, 0, 2])
ax4.set_xlabel(r"$\omega$ / rad ps$^{-1}$")
ax4.set_ylabel(r"$\omega$ / rad ps$^{-1}$")

for i, ax in enumerate([ax1, ax2, ax3, ax4]):
    fp.panel_label(ax, i)

plt.savefig(paths.figures / "water_acf.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close()
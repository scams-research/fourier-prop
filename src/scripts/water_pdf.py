import numpy as np
import scipp as sc
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
from MDAnalysis import transformations
from MDAnalysis.coordinates.memory import MemoryReader
import matplotlib.pyplot as plt
from tqdm import tqdm

from fourierror import numerical
from fourierror.data_array import CovDataArray

import paths
import _fig_params as fp

data = paths.data / 'tip4p_hold.data'
traj = paths.data / 'tip4p_hold.atom'

u = mda.Universe(data, traj, format='LAMMPSDUMP', time_step=1.0, time_unit='ps')

FRAC = 0.05

oxygen   = u.select_atoms('type 1')
hydrogen = u.select_atoms('type 2')
assert len(hydrogen) == 2*len(oxygen), "expected 2 H per O"

N_SNAP = 200
NBINS = 150
N_SAMPLES = 5000
RRANGE = (0.05, 15.05)

L = u.trajectory[0].dimensions[:3].min()
assert RRANGE[1] <= L/2, f"r_max {RRANGE[1]} exceeds L/2 = {L/2:.2f} A"
 
frames = np.linspace(0, len(u.trajectory)-1, N_SNAP, dtype=int)
 
OO = np.zeros((N_SNAP, NBINS))
OH = np.zeros((N_SNAP, NBINS))
HH = np.zeros((N_SNAP, NBINS))

rng = np.random.default_rng(42)

# Densities and concentrations stay FULL-SYSTEM — subsampling is an
# estimator for g(r), it does not change the physical density.
RHO = u.atoms.n_atoms / u.trajectory[0].volume
CONC = {"O": len(oxygen)/u.atoms.n_atoms,
        "H": len(hydrogen)/u.atoms.n_atoms}

n_mol = u.residues.n_residues
n_pick = max(2, int(FRAC * n_mol))

for j, i in enumerate(tqdm(frames)):
    # redraw each snapshot; keep molecules whole and in index order
    pick = np.sort(rng.choice(n_mol, size=n_pick, replace=False))
    sub = u.residues[pick].atoms
    O_sub = sub.select_atoms('type 1')
    H_sub = sub.select_atoms('type 2')

    a = rdf.InterRDF(O_sub, O_sub, range=RRANGE, nbins=NBINS, exclusion_block=(1,1))
    a.run(start=i, stop=i+1); OO[j] = a.results.rdf

    c = rdf.InterRDF(H_sub, H_sub, range=RRANGE, nbins=NBINS, exclusion_block=(2,2))
    c.run(start=i, stop=i+1); HH[j] = c.results.rdf
 
r = a.results.bins

def faberzimen(r, snapshots, rho):
    """A_ij(Q) and its covariance, from an (n_snapshots, nbins) array."""
    y = snapshots.mean(0)
 
    n = len(snapshots)
    cov = np.cov(snapshots.T) / n
 
    dr = r[1] - r[0]
    r0 = r[0]                      
 
    r_data = sc.array(dims=['r'], values=r, unit='angstrom')
    y_data = CovDataArray(data=sc.array(dims=['r'], values=y),
                          coords={'r': r_data}, covariance=cov)
    # The sine transform, sum_n f(r_n) sin(Q r_n) dr, equals -Im of the FT
    # only on a grid starting at r = 0. The first bin is at r0 = k dr, so
    # prepend k points at r = 0, ..., (k - 1) dr, where r [g(r) - 1] is
    # identically zero (g = 0 below the closest approach, and r = 0 at r = 0).
    k = int(round(r0 / dr))
    assert np.isclose(r0, k * dr), "first bin must lie on the r = n dr grid"
    n_r = len(r) + k
    r_pad = np.concatenate([np.arange(k) * dr, r])
    y_pad = np.concatenate([np.zeros(k), y])
    cov_pad = np.zeros((n_r, n_r))
    cov_pad[k:, k:] = cov

    r_pad_data = sc.array(dims=['r'], values=r_pad, unit='angstrom')
    f_data = r_pad_data * (CovDataArray(data=sc.array(dims=['r'], values=y_pad),
                                        coords={'r': r_pad_data}, covariance=cov_pad) - 1)
    ff = numerical.dft(f_data, 'r', N_SAMPLES, random_state=rng)
    Q = ff['imag'].coords['omega'].values

    # numerical.dft returns (2 / n) * FFT, so undo that normalisation.
    scale = n_r / 2.0 * dr
    sine = -ff['imag'].values * scale
    cov_sine = ff['imag'].covariance.values * scale ** 2

    pref = 4*np.pi*rho/Q
    A = 1.0 + pref*sine

    covA = cov_sine * np.outer(pref, pref)
    
    return CovDataArray(data=sc.array(dims=['q'], values=A), coords={'q': sc.array(dims=['q'], values=Q, unit='1/angstrom')}, covariance=covA), y_data

A_OO, R_OO = faberzimen(r, OO, RHO)
A_HH, R_HH = faberzimen(r, HH, RHO)

def plot_spectrum(ds, ax, label, coord, offset=0, **kw):
    """Plot a spectrum with a 95 % confidence band (1.96 sigma)."""
    e = np.sqrt(ds.variances)
    w = ds.coords[coord].values
    v = ds.values
    ax.plot(w, v+offset, label=label, **kw)
    ax.fill_between(w, (v + offset) - 1.96 * e, (v + offset) + 1.96 * e, alpha=0.25, lw=0, **kw)

fig = plt.figure(figsize=(fp.COLUMN_WIDTH, 3.0))
gs = plt.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.35, height_ratios=[1, 0.5])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1]) 
ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax4 = fig.add_subplot(gs[1, 1], sharex=ax2)

plot_spectrum(R_HH, ax1, 'H-H', 'r', color=fp.colors[0])
plot_spectrum(R_OO, ax1, 'O-O', 'r', offset=2, color=fp.colors[1])
ax1.set(xlabel=r"$r$ / Å", ylabel="$g(r)$")
ax1.set_xlim(0, None)
ax1.set_xticks([0, 4, 8, 12])
ax1.set_ylim(0, None)
ax1.set_yticks([0, 2, 4])
ax1.legend()

mask_OO = A_OO.coords['q'] > 0 * sc.Unit('1/angstrom')
mask_HH = A_HH.coords['q'] > 0 * sc.Unit('1/angstrom')
plot_spectrum(A_HH[mask_HH], ax2, "", 'q', color=fp.colors[0])
plot_spectrum(A_OO[mask_OO], ax2, "", 'q', offset=3, color=fp.colors[1])
ax2.set(xlabel=r"$q$ / Å$^{-1}$", ylabel="$A(q)$")
ax2.set_xlim(0, None)
ax2.set_xticks([0, 10, 20, 30])
ax2.set_ylim(0, None)
ax2.set_yticks([0, 2, 4])

ax3.plot(R_HH.coords['r'].values, R_HH.variances, color=fp.colors[0])
ax3.plot(R_OO.coords['r'].values, R_OO.variances, color=fp.colors[1])
ax3.set(xlabel=r"$r$ / Å", ylabel=r"$\sigma^2[g(r)]$", yscale="log")

ax4.plot(A_HH[mask_HH].coords['q'].values, A_HH[mask_HH].variances, color=fp.colors[0])
ax4.plot(A_OO[mask_OO].coords['q'].values, A_OO[mask_OO].variances, color=fp.colors[1])
ax4.set(xlabel=r"$q$ / Å$^{-1}$", ylabel=r"$\sigma^2[A(q)]$", yscale="log")

for i, ax in enumerate([ax1, ax2, ax3, ax4]):
    fp.panel_label(ax, i)

plt.savefig(paths.figures / "water_pdf.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close()

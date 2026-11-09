from matplotlib import rcParams
import matplotlib.font_manager as fm
helvetica = False
try:
    font_files = fm.findSystemFonts("/Users/ee23687/workspace/other/fonts", fontext="ttf")
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    helvetica = len(font_files) > 0
except Exception as e:
    print("Warning: Could not load Helvetica font.")
    print(f"{e}")
    print("Falling back to default sans-serif font.")


# blue, green, orange, pink, yellow, light blue
colors = ["#0173B2", "#029E73", "#D55E00", "#CC78BC", "#ECE133", "#56B4E9"]
FONTSIZE = 6
NEARLY_BLACK = "#161616"
LIGHT_GREY = "#F5F5F5"
WHITE = "#ffffff"
NBINS = 25
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
CREDIBLE_INTERVALS = [[15.9, 84.1, 0.6], [2.3, 97.7, 0.3], [0.15, 99.85, 0.1]]

MASTER_FORMATTING = {
    "axes.formatter.limits": (-3, 3),
    "xtick.major.pad": 1,
    "ytick.major.pad": 1,
    "ytick.color": NEARLY_BLACK,
    "xtick.color": NEARLY_BLACK,
    "axes.labelcolor": NEARLY_BLACK,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.0,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.handletextpad": 0.4,
    'axes.edgecolor': NEARLY_BLACK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # "mathtext.fontset": "stixsans",
    "mathtext.fontset": "custom",
    "mathtext.bf": "sans:bold",
    "mathtext.bfit": "sans:bold:italic",
    "mathtext.it": "sans:italic",
    "mathtext.rm": "sans",
    "mathtext.cal": "cmsy10",
    "mathtext.default": "it",
    "font.size": FONTSIZE,
    "font.family": "sans-serif",
    "savefig.bbox": "tight",
    "axes.facecolor": WHITE,
    "axes.labelpad": 1.0,
    "axes.labelsize": FONTSIZE,
    "axes.titlepad": 8,
    "axes.titlesize": FONTSIZE,
    "axes.grid": False,
    "grid.color": WHITE,
    "lines.markersize": 1.0,
    "xtick.major.size": 2.0,
    "xtick.major.width": 1.0,
    "ytick.major.size": 2.0,
    "ytick.major.width": 1.0,
    "lines.scale_dashes": False,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE,
    "lines.linewidth": 1,
}
if helvetica:
    MASTER_FORMATTING["font.family"] = "Helvetica"
    MASTER_FORMATTING["font.sans-serif"] = ["Helvetica"]

for k, v in MASTER_FORMATTING.items():
    rcParams[k] = v

# 510.0pt. \textwidth
# 246.0pt. \columnwidth

COLUMN_WIDTH = 246.0 / 72.27  # inches
PAGE_WIDTH = 510.0 / 72.27  # inches
GOLDEN_RATIO = (1 + 5**0.5) / 2  # Approx. 1.618

def panel_label(ax, i):
    """Add the panel letter at the top-left of an axes, as in every figure."""
    ax.text(0.01, 1.02, ALPHABET[i], transform=ax.transAxes, ha="center", va="bottom")


DIVERGING_CMAP = "RdBu_r"


def matrix_map(fig, ax, x, y, z, label):
    """Draw a (co)variance matrix on a symmetric-log diverging colour scale.

    The scale is linear within +/- `linthresh` of zero and logarithmic beyond,
    so small off-diagonal elements remain visible alongside a dominant
    diagonal. `linthresh` is the 95th percentile of the absolute off-diagonal
    elements, so that sampling noise stays in the faint linear region. Values
    are scaled by a power of ten, which is given in the colour bar label. The
    colour bar sits to the right of the axes, leaving the top-left free for
    the panel letter.
    """
    import numpy as np
    from matplotlib.colors import SymLogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    z = np.asarray(z)
    exponent = int(np.floor(np.log10(np.abs(z).max())))
    zs = z / 10 ** exponent
    lim = np.abs(zs).max()
    off = np.abs(zs[~np.eye(zs.shape[0], dtype=bool)])
    linthresh = np.percentile(off, 95)
    norm = SymLogNorm(linthresh=linthresh, linscale=0.5, vmin=-lim, vmax=lim, base=10)
    im = ax.pcolormesh(x, y, zs, cmap=DIVERGING_CMAP, norm=norm,
                       shading="nearest", rasterized=True)
    ax.set_aspect("equal")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    # Ticks at zero and at most three decades either side, above linthresh.
    decades = np.arange(np.ceil(np.log10(linthresh)), np.floor(np.log10(lim)) + 1)
    if decades.size == 0:
        decades = np.array([np.floor(np.log10(lim))])
    decades = decades[-3:]
    ticks = [-(10.0 ** d) for d in decades[::-1]] + [0] + [10.0 ** d for d in decades]
    labels = ([rf"$-10^{{{int(d)}}}$" for d in decades[::-1]] + ["0"]
              + [rf"$10^{{{int(d)}}}$" for d in decades])
    cbar.set_ticks(ticks, labels=labels)
    cbar.ax.minorticks_off()
    if exponent != 0:
        label = label + rf" / $10^{{{exponent}}}$"
    cbar.set_label(label)
    return im, cbar

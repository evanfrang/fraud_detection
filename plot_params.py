import matplotlib as mpl # type: ignore
import matplotlib.pyplot as plt # type: ignore

plt.style.use('bmh')

CUSTOM_RCPARAMS = {
    "text.usetex": True,
    "axes.linewidth": 2,
    "font.family": "serif",
    "font.weight": "bold",
    "text.latex.preamble": r"\usepackage{sfmath} \boldmath",
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 22,
    "legend.fontsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 16,
}

# Function to apply settings
def apply_rcparams():
    mpl.rcParams.update(CUSTOM_RCPARAMS)
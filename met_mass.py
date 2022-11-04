import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import matplotlib.gridspec as gridspec
import eagle_IO.eagle_IO as eagle_io
from flare import plt as flareplt
from unyt import mh, cm, Gyr, g, Msun, Mpc
from utils import mkdir, plot_meidan_stat, get_nonmaster_evo_data
from utils import get_nonmaster_centred_data, grav_enclosed, calc_ages
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Planck18 as cosmo, z_at_value
from scipy.spatial import cKDTree
import cmasher as cmr


def plot_met_mass_relation_vary(snap):

    # Define redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Define the path
    ini_path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

    # Define physics variations directories
    types = ["flares_00", "FLARES_00_REF",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh",
             "flares_00_no_agn", "FLARES_00_highFBlim",
             "FLARES_00_medFBlim", "FLARES_00_slightFBlim",
             "flares_00_H_reion_z03",
             "flares_00_H_reion_z075", "flares_00_H_reion_z14"]

    # Define labels for each
    labels = ["AGNdT9", "REF", "SKIP",
              "InstantFB", "$Z^0$", "$M_\dot=0$",
              "$f_{\mathrm{th, max}}=10$", "$f_{\mathrm{th, max}}=6$",
              "$f_{\mathrm{th, max}}=4$",
              "$z_{\mathrm{reion}}=3.0$", "$z_{\mathrm{reion}}=7.5$",
              "$z_{\mathrm{reion}}=14.0$"]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted", "dotted",
                  "dashdot", "dashdot", "dashdot"]

    # Define plot dimensions
    nrows = 4
    ncols = 3

    # Define norm
    norm = LogNorm(vmin=1, vmax=10)

    # Define hexbin extent
    extent = [8, 11.5, -2, 1.5]

    # Set up the plots
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.5))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[1:, -1])

    for i in range(nrows):
        for j in range(ncols):

            if i * ncols + j >= len(labels):
                continue

            if labels[i * ncols + j] == "SKIP":
                continue

            # Create axis
            ax = fig.add_subplot(gs[i, j])
            ax.grid(True)

            # Include labels
            if j == 0:
                ax.set_ylabel(r"$Z$")
            if i == nrows - 1:
                ax.set_xlabel(r"$M_\star / M_\odot$")

            # Remove unnecessary ticks
            if j > 0:
                ax.tick_params("y", left=False, right=False,
                               labelleft=False, labelright=False)

            if i < nrows - 1:
                ax.tick_params("x", top=False, bottom=False,
                               labeltop=False, labelbottom=False)

            # Set axis limits
            ax.set_ylim(10**extent[2], 10**extent[3])
            ax.set_xlim(10**extent[0], 10**extent[1])

            # Label axis
            ax.text(0.95, 0.9, labels[i * ncols + j],
                    bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                              alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            axes.append(ax)

    for (ind, t), l in zip(enumerate(types), labels):

        path = ini_path.replace("<type>", t)

        print(path)

        mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/ApertureMeasurements/Mass/030kpc",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)[:, 4] * 10 ** 10
        met = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                  snap,
                                  "Subhalo/Stars/SmoothedMetallicity",
                                  noH=True, physicalUnits=True,
                                  numThreads=8)[:, 4] * 10 ** 3

        # Apply some cuts
        okinds = mass > 10**8

        im = axes[ind].hexbin(mass[okinds], met[okinds], mincnt=1, gridsize=50,
                              xscale="log", yscale="log", linewidth=0.2,
                              cmap="plasma", norm=norm, extent=extent)

    # Set up colorbar
    cbar = fig.colorbar(im, cax)
    cbar.set_label("$N$")

    # Save figure
    mkdir("plots/massmet/")
    fig.savefig("plots/massmet/mass_metallicity_%s.png" % snap,
                bbox_inches="tight")


if __name__ == "__main__":
    plot_met_mass_relation_vary("010_z005p000")

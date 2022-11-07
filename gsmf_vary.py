import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, TwoSlopeNorm
import matplotlib.gridspec as gridspec
import eagle_IO.eagle_IO as eagle_io
from flare import plt as flareplt
from unyt import mh, cm, Gyr, g, Msun, Mpc
from utils import mkdir, plot_meidan_stat, get_nonmaster_evo_data, mass_bins
from utils import get_nonmaster_centred_data, grav_enclosed, calc_ages
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Planck18 as cosmo, z_at_value
from scipy.spatial import cKDTree
import cmasher as cmr

import fitDF.fitDF as fitDF
import fitDF.models as models
import fitDF.analyse as analyse


# Turn on grid
mpl.rcParams.update({"axes.grid": True})


def plot_df(self, ax, phi, phi_sigma, hist, massBins,
            label, color, hist_lim=10, lw=3, alpha=0.7, lines=True):

    kwargs = {}
    kwargs_lo = {}

    if lines:
        kwargs['lw'] = lw

        kwargs_lo['lw'] = lw
        kwargs_lo['linestyle'] = 'dotted'
    else:
        kwargs['ls'] = ''
        kwargs['marker'] = 'o'

        kwargs_lo['ls'] = ''
        kwargs_lo['marker'] = 'o'
        kwargs_lo['markerfacecolor'] = 'white'
        kwargs_lo['markeredgecolor'] = color

    def yerr(phi, phi_sigma):

        p = phi
        ps = phi_sigma

        mask = (ps == p)

        err_up = np.abs(np.log10(p) - np.log10(p + ps))
        err_lo = np.abs(np.log10(p) - np.log10(p - ps))

        err_lo[mask] = 100

        return err_up, err_lo, mask

    err_up, err_lo, mask = yerr(phi, phi_sigma)

    # err_lo = np.log10(phi) - np.log10(phi - phi_sigma[0])
    # err_up = np.log10(phi) - np.log10(phi + phi_sigma[1])

    ax.errorbar(np.log10(massBins[phi > 0.]),
                np.log10(phi[phi > 0.]),
                yerr=[err_lo[phi > 0.],
                      err_up[phi > 0.]],
                # uplims=(mask[phi > 0.]),
                label=label, alpha=alpha, **kwargs)


def plot_gsmf_evo_vary():

    # Set up model
    model = models.DoubleSchechter()

    # Define binning
    massBins, massBinLimits = mass_bins()

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
    labels = ["AGNdT9", "REF",
              "InstantFB", "$Z^0$", "$M_\dot=0$",
              "$f_{\mathrm{th, max}}=10$", "$f_{\mathrm{th, max}}=6$",
              "$f_{\mathrm{th, max}}=4$",
              "$z_{r, 0}$", "$z_{r, 7.5}$",
              "$z_{r, 14}$"]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted", "dotted",
                  "dashdot", "dashdot", "dashdot"]

    # Define snapshots
    snaps = ["009_z006p000", "007_z008p000", "008_z007p000",
             "009_z006p000", "010_z005p000"]

    # Define plot dimensions
    nrows = 1
    ncols = len(snaps)

    # Define norm
    norm = LogNorm(vmin=1, vmax=10)

    # Define hexbin extent
    extent = [8, 11.5, 0, 2]

    # Set up the plots
    fig = plt.figure(figsize=(ncols * 2.5, nrows * 2.5))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[-1])

    for j in range(ncols):

        # Define redshift
        z = float(snaps[j].split("z")[-1].replace("p", "."))

        # Create axis
        ax = fig.add_subplot(gs[j])

        # Include labels
        if j == 0:
            ax.set_ylabel(
                '$\mathrm{log_{10}}\,(\phi \,/\, \mathrm{Mpc^{-3}} \, \mathrm{dex^{-1}})$')
        ax.set_xlabel(
            '$\mathrm{log_{10}} \, (M_{*} \,/\, \mathrm{M_{\odot}})$')

        # Remove unnecessary ticks
        if j > 0:
            ax.tick_params("y", left=False, right=False,
                           labelleft=False, labelright=False)

        # Set axis limits
        ax.set_ylim(10**extent[2], 10**extent[3])
        ax.set_xlim(10**extent[0], 10**extent[1])

        # Label axis
        ax.text(0.95, 0.9, "$z=$%d" % z,
                bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                          alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right',
                fontsize=8)

        axes.append(ax)

    for ax, tag in zip(axes, snaps):

        for (ind, t), l in zip(enumerate(types), labels):

            path = ini_path.replace("<type>", t)

            print(path)

            # Get the arrays from the raw data files
            try:
                mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                           tag,
                                           "Subhalo/ApertureMeasurements/Mass/030kpc",
                                           noH=True, physicalUnits=True,
                                           numThreads=8)[:, 4] * 10 ** 10
            except ValueError:
                continue

            okinds = mass > 0
            mass = mass[okinds]

            V = (4./3) * np.pi * (0.03)**3

            hist, dummy = np.histogram(np.log10(mass), bins=massBinLimits)
            hist = np.float64(hist)
            phi = (hist / V) / (massBinLimits[1] - massBinLimits[0])

            phi_sigma = (np.sqrt(hist) / V) / \
                (massBinLimits[1] - massBinLimits[0])

            phi_all = np.array(phi)
            hist_all = hist

            phi_sigma = np.sqrt(np.sum(np.square(phi_sigma), axis=0))

            # # ---- Get fit
            # sample_ID = 'flares_gsmf_%s' % (tag)

            # a = analyse.analyse(ID='samples', model=model,
            #                     sample_save_ID=None, verbose=False)

            # if 'color' in ax._get_lines._prop_keys:
            #     c = next(ax._get_lines.prop_cycler)['color']

            plot_df(ax, phi_all, phi_sigma, hist_all,
                    massBins=massBins, color=None, lines=linestyles[ind], label=l)
            # model.update_params(a.median_fit)

            # xvals = np.linspace(7, 15, 1000)
            # ax.plot(xvals, a.model.log10phi(xvals), color=c, label=l)

    # Draw legend
    axes[2].legend(loc='upper center',
                   bbox_to_anchor=(0.5, -0.2),
                   fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/gsmf/")
    fig.savefig("plots/gsmf/gsmf.png",
                bbox_inches="tight")


if __name__ == "__main__":
    plot_gsmf_evo_vary()

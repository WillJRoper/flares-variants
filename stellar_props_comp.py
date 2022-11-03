import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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


# Turn on grid
mpl.rcParams.update({"axes.grid": True})


def plot_birth_density_evo():

    flares_z_bins = np.arange(4.5, 15.5, 1.0)

    # Define the path
    path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

   # Define physics variations directories
    types = ["flares_00", "FLARES_00_REF",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh",
             "flares_00_no_agn", "FLARES_00_highFBlim",
             "FLARES_00_medFBlim", "FLARES_00_slightFBlim",
             "flares_00_H_reion_z03",
             "flares_00_H_reion_z075", "flares_00_H_reion_z14"]
    types = types[::-1]

    # Define labels for each
    labels = ["AGNdT9", "REF",
              "InstantFB", "$Z^0$", "$M_\dot=0$",
              "$f_{\mathrm{th, max}}=10$", "$f_{\mathrm{th, max}}=6$",
              "$f_{\mathrm{th, max}}=4$",
              "$z_{\mathrm{reion}}=3.0$", "$z_{\mathrm{reion}}=7.5$",
              "$z_{\mathrm{reion}}=14.0$"]
    labels = labels[::-1]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted", "dotted",
                  "dashdot", "dashdot", "dashdot"]
    linestyles = linestyles[::-1]

    # Define snapshot for the root
    snap = "011_z004p770"

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Log the y axis
    ax.semilogy()

    # Loop over the variants
    for t, l, ls in zip(types, labels, linestyles):

        # Get the arrays from the raw data files
        aborn = eagle_io.read_array('PARTDATA', path.replace("<type>", t),
                                    snap,
                                    'PartType4/StellarFormationTime',
                                    noH=True, physicalUnits=True,
                                    numThreads=8)
        den_born = (eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                        snap, "PartType4/BirthDensity",
                                        noH=True, physicalUnits=True,
                                        numThreads=8) * 10**10
                    * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value

        # Convert to redshift
        zs = 1 / aborn - 1

        # Plot median curves
        plot_meidan_stat(zs, den_born, np.ones(den_born.size), ax,
                         lab=l, bins=flares_z_bins, color=None, ls=ls)

    # Label axes
    ax.set_ylabel(r"$n_{\mathrm{H}} / \mathrm{cm}^{-3}$")
    ax.set_xlabel(r"$z_{\mathrm{birth}}$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/stellar_birthden_%s.png" % snap,
                bbox_inches="tight")


def plot_birth_met_evo():

    flares_z_bins = np.arange(4.5, 15.5, 1.0)

    # Define the path
    path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

   # Define physics variations directories
    types = ["flares_00", "FLARES_00_REF",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh",
             "flares_00_no_agn", "FLARES_00_highFBlim",
             "FLARES_00_medFBlim", "FLARES_00_slightFBlim",
             "flares_00_H_reion_z03",
             "flares_00_H_reion_z075", "flares_00_H_reion_z14"]
    types = types[::-1]

    # Define labels for each
    labels = ["AGNdT9", "REF",
              "InstantFB", "$Z^0$", "$M_\dot=0$",
              "$f_{\mathrm{th, max}}=10$", "$f_{\mathrm{th, max}}=6$",
              "$f_{\mathrm{th, max}}=4$",
              "$z_{\mathrm{reion}}=3.0$", "$z_{\mathrm{reion}}=7.5$",
              "$z_{\mathrm{reion}}=14.0$"]
    labels = labels[::-1]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted", "dotted",
                  "dashdot", "dashdot", "dashdot"]
    linestyles = linestyles[::-1]

    # Define snapshot for the root
    snap = "011_z004p770"

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Log the y axis
    ax.semilogy()

    # Loop over the variants
    for t, l, ls in zip(types, labels, linestyles):

        # Get the arrays from the raw data files
        aborn = eagle_io.read_array('PARTDATA', path.replace("<type>", t),
                                    snap,
                                    'PartType4/StellarFormationTime',
                                    noH=True, physicalUnits=True,
                                    numThreads=8)
        met = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                  snap, "PartType4/SmoothedMetallicity",
                                  noH=True, physicalUnits=True,
                                  numThreads=8)

        # Convert to redshift
        zs = 1 / aborn - 1

        # Plot median curves
        plot_meidan_stat(zs, met, np.ones(met.size), ax,
                         lab=l, color=None, bins=flares_z_bins, ls=ls)

    # Label axes
    ax.set_ylabel(r"$Z_{\mathrm{birth}}$")
    ax.set_xlabel(r"$z_{\mathrm{birth}}$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/stellar_met_%s.png" % snap,
                bbox_inches="tight")


def plot_birth_met_vary(snap):

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

    # Define plot dimensions
    nrows = 4
    ncols = 3

    # Define norm
    norm = LogNorm(vmin=1, vmax=50000)
    resi_norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    outlier_norm = TwoSlopeNorm(vmin=-10.01, vcenter=0, vmax=10)

    # Define hexbin extent
    extent = [4.6, 25, 0, 0.119]

    # Set up the plot
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.5))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[-1, -1])

    for i in range(nrows):
        for j in range(ncols):

            if i * ncols + j >= len(labels):
                continue

            if labels[i * ncols + j] == "SKIP":
                continue

            # Create axis
            ax = fig.add_subplot(gs[i, j])

            # Include labels
            if j == 0:
                ax.set_ylabel(r"$Z_{\mathrm{birth}}$")
            if i == nrows - 1:
                ax.set_xlabel(r"$z_{\mathrm{birth}}$")

            # Remove unnecessary ticks
            if j > 0:
                ax.tick_params("y", left=False, right=False,
                               labelleft=False, labelright=False)
            if i < nrows - 1:
                ax.tick_params("x", top=False, bottom=False,
                               labeltop=False, labelbottom=False)

            # Set axis limits
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlim(extent[0], extent[1])

            # Label axis
            ax.text(0.95, 0.9, labels[i * ncols + j],
                    bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                              alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            axes.append(ax)

    # Initialise a dictionary to store the hexbins
    hex_dict = {}

    for (ind, t), l in zip(enumerate(types), labels):

        path = ini_path.replace("<type>", t)

        print(path)

        reg_zs, reg_mets = get_nonmaster_evo_data(
            path, snap, y_key="PartType4/SmoothedMetallicity")

        im = axes[ind].hexbin(reg_zs, reg_mets, mincnt=0, gridsize=30,
                              linewidth=0.2, cmap="plasma",
                              norm=norm, extent=extent)

        hex_dict[t] = {"zs": reg_zs, "mets": reg_mets, "h": im.get_array()}

    # Set up colorbar
    cbar = fig.colorbar(im, cax)
    cbar.set_label("$N$")

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/stellar_birthZ_%s.png" % snap,
                bbox_inches="tight")
    plt.close(fig)

    # Define labels for each
    labels = ["AGNdT9", "REF", "SKIP",
              "InstantFB", "$Z^0$", "$M_\dot=0$",
              "$f_{\mathrm{th, max}}=10$", "$f_{\mathrm{th, max}}=6$",
              "$f_{\mathrm{th, max}}=4$",
              "$z_{\mathrm{reion}}=3.0$", "$z_{\mathrm{reion}}=7.5$",
              "$z_{\mathrm{reion}}=14.0$"]

    # Set up the plot
    fig = plt.figure(figsize=(len(labels) * 2.5, len(labels) * 2.5))
    gs = gridspec.GridSpec(nrows=len(labels) + 1, ncols=len(labels) + 1,
                           width_ratios=[20, ] * len(labels) + [1, ],
                           height_ratios=[1, ] + [20, ] * len(labels))
    gs.update(wspace=0.0, hspace=0.0)
    axes = np.zeros((len(labels), len(labels)), dtype=object)
    cax1 = fig.add_subplot(gs[:, -1])
    cax2 = fig.add_subplot(gs[0, :])

    # Loop over models and construct corner plot
    for i, ti in enumerate(types):
        for j, tj in enumerate(types):

            # Create axis
            ax = fig.add_subplot(gs[i + 1, j])

            # Include labels
            if j == 0:
                ax.set_ylabel(r"$Z_{\mathrm{birth}}$")
            if i == len(labels) - 1:
                ax.set_xlabel(r"$z_{\mathrm{birth}}$")

            # Remove unnecessary ticks
            if j > 0:
                ax.tick_params("y", left=False, right=False,
                               labelleft=False, labelright=False)
            if i < len(labels) - 1:
                ax.tick_params("x", top=False, bottom=False,
                               labeltop=False, labelbottom=False)

            # Set axis limits
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlim(extent[0], extent[1])

            # Label axis
            if j == 0:
                ax.text(-0.5, 0.5, labels[i],
                        transform=ax.transAxes, verticalalignment='center',
                        fontsize=12, rotation=90)
            if i == len(labels) - 1:
                ax.text(0.5, -0.3, labels[j],
                        transform=ax.transAxes, horizontalalignment='center',
                        fontsize=12)

            axes[i, j] = ax

            if j == i:
                im = axes[i, j].hexbin(hex_dict[ti]["zs"],
                                       hex_dict[ti]["mets"],
                                       mincnt=0, gridsize=30, linewidth=0.2,
                                       norm=norm,  cmap="plasma", extent=extent)

                # Set up colorbar
                cbar = fig.colorbar(im, cax1)
                cbar.set_label("$N$")
            else:
                im = axes[i, j].hexbin(hex_dict[ti]["zs"],
                                       hex_dict[ti]["mets"],
                                       gridsize=30, linewidth=0.2,
                                       cmap="cmr.guppy",
                                       extent=extent)
                hi = hex_dict[ti]["h"]
                hj = hex_dict[tj]["h"]
                hokinds = np.logical_and(hi == 0,
                                         hj == 0)
                new_arr = np.log10((hi / np.sum(hi)) / (hj / np.sum(hj)))
                bkg_arr = np.full_like(new_arr, np.nan)
                print(np.min(new_arr), np.max(new_arr))
                hi_okinds = np.logical_and(hi > 0, hj == 0)
                hj_okinds = np.logical_and(hi == 0, hj > 0)
                new_arr[hokinds] = np.nan
                bkg_arr[hi_okinds] = 10
                bkg_arr[hj_okinds] = -10
                im.set_array(new_arr)
                im.set_norm(resi_norm)

                im1 = axes[i, j].hexbin(hex_dict[ti]["zs"],
                                        hex_dict[ti]["mets"],
                                        gridsize=20, linewidth=0.2,
                                        cmap="cmr.guppy",
                                        extent=extent, alpha=0.2)
                im1.set_array(bkg_arr)
                im1.set_norm(resi_norm)

                # Set up colorbar
                cbar = fig.colorbar(im, cax2, orientation="horizontal")
                cbar.set_label("$\log_{10}(P_{i} / P_{j})$")
                cbar.ax.xaxis.set_ticks_position('top')
                cbar.ax.xaxis.set_label_position('top')
                cbar.ax.xaxis.set_ticks([-2, -1, 0, 1, 2])
                cbar.ax.xaxis.set_ticklabels(
                    ["$\leq-2$", "-1", "0", "1", "$2\leq$"])

    fig.savefig("plots/physics_vary/stellar_birthZ_residual.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_birth_den_vary(snap):

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

    # Define plot dimensions
    nrows = 4
    ncols = 3

    # Define norm
    norm = LogNorm(vmin=1, vmax=50000)
    resi_norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    outlier_norm = TwoSlopeNorm(vmin=-10.01, vcenter=0, vmax=10)

    # Define hexbin extent
    extent = [4.6, 22, -2.2, 5.5]

    # Set up the plot
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.5))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[-1, -1])

    for i in range(nrows):
        for j in range(ncols):

            if i * ncols + j >= len(labels):
                continue

            if labels[i * ncols + j] == "SKIP":
                continue

            # Create axis
            ax = fig.add_subplot(gs[i, j])
            ax.semilogy()

            # Include labels
            if j == 0:
                ax.set_ylabel(r"$n_{\mathrm{H}} / \mathrm{cm}^{-3}$")
            if i == nrows - 1:
                ax.set_xlabel(r"$z_{\mathrm{birth}}$")

            # Remove unnecessary ticks
            if j > 0:
                ax.tick_params("y", left=False, right=False,
                               labelleft=False, labelright=False)
            if i < nrows - 1:
                ax.tick_params("x", top=False, bottom=False,
                               labeltop=False, labelbottom=False)

            # Set axis limits
            ax.set_ylim(10**extent[2], 10**extent[3])
            ax.set_xlim(extent[0], extent[1])

            # Label axis
            ax.text(0.95, 0.9, labels[i * ncols + j],
                    bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                              alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            axes.append(ax)

    # Initialise a dictionary to store the hexbins
    hex_dict = {}

    for (ind, t), l in zip(enumerate(types), labels):

        path = ini_path.replace("<type>", t)

        print(path)

        reg_zs, reg_dens = get_nonmaster_evo_data(
            path, snap, y_key="PartType4/BirthDensity")

        # Convert density to hydrogen number density
        reg_dens = (reg_dens * 10**10
                    * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value

        im = axes[ind].hexbin(reg_zs, reg_dens, mincnt=0, gridsize=20,
                              yscale="log", linewidth=0.2, cmap="plasma",
                              extent=extent)

        hex_dict[t] = {"zs": reg_zs, "dens": reg_dens, "h": im.get_array()}

    # Set up colorbar
    cbar = fig.colorbar(im, cax)
    cbar.set_label("$N$")

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/stellar_birthden_%s.png" % snap,
                bbox_inches="tight")
    plt.close(fig)

    # Define labels for each
    labels = ["AGNdT9", "REF",
              "InstantFB", "$Z^0$", "$M_\dot=0$",
              "$f_{\mathrm{th, max}}=10$", "$f_{\mathrm{th, max}}=6$",
              "$f_{\mathrm{th, max}}=4$",
              "$z_{\mathrm{reion}}=3.0$", "$z_{\mathrm{reion}}=7.5$",
              "$z_{\mathrm{reion}}=14.0$"]

    # Set up the plot
    fig = plt.figure(figsize=(len(labels) * 1.0, len(labels) * 1.0))
    gs = gridspec.GridSpec(nrows=len(labels) + 1, ncols=len(labels) + 1,
                           width_ratios=[20, ] * len(labels) + [1, ],
                           height_ratios=[1, ] + [20, ] * len(labels))
    gs.update(wspace=0.0, hspace=0.0)
    axes = np.zeros((len(labels), len(labels)), dtype=object)
    cax1 = fig.add_subplot(gs[:, -1])
    cax2 = fig.add_subplot(gs[0, :])

    # Loop over models and construct corner plot
    for i, ti in enumerate(types):
        for j, tj in enumerate(types):

            # Create axis
            ax = fig.add_subplot(gs[i + 1, j])

            # Include labels
            if j == 0:
                ax.set_ylabel(r"$n_{\mathrm{H}} / \mathrm{cm}^{-3}$")
            if i == len(labels) - 1:
                ax.set_xlabel(r"$z_{\mathrm{birth}}$")

            # Remove unnecessary ticks
            if j > 0:
                ax.tick_params("y", left=False, right=False,
                               labelleft=False, labelright=False)
            if i < len(labels) - 1:
                ax.tick_params("x", top=False, bottom=False,
                               labeltop=False, labelbottom=False)

            # Set axis limits
            ax.set_ylim(10**extent[2], 10**extent[3])
            ax.set_xlim(extent[0], extent[1])

            # Label axis
            if j == 0:
                ax.text(-1.0, 0.5, labels[i],
                        transform=ax.transAxes, verticalalignment='center',
                        rotation=90)
            if i == len(labels) - 1:
                ax.text(0.5, -1.0, labels[j],
                        transform=ax.transAxes, horizontalalignment='center')

            axes[i, j] = ax

            if j == i:
                im = axes[i, j].hexbin(hex_dict[ti]["zs"],
                                       hex_dict[ti]["dens"],
                                       yscale="log",
                                       mincnt=0, gridsize=20, linewidth=0.2,
                                       norm=norm,  cmap="plasma", extent=extent)

                # Set up colorbar
                cbar = fig.colorbar(im, cax1)
                cbar.set_label("$N$")
            else:
                im = axes[i, j].hexbin(hex_dict[ti]["zs"],
                                       hex_dict[ti]["dens"],
                                       gridsize=20, linewidth=0.2,
                                       yscale="log",
                                       cmap="cmr.guppy",
                                       extent=extent)
                hi = hex_dict[ti]["h"]
                hj = hex_dict[tj]["h"]
                hokinds = np.logical_and(hi == 0,
                                         hj == 0)
                new_arr = np.log10((hi / np.sum(hi)) / (hj / np.sum(hj)))
                bkg_arr = np.full_like(new_arr, np.nan)
                print(np.min(new_arr), np.max(new_arr))
                hi_okinds = np.logical_and(hi > 0, hj == 0)
                hj_okinds = np.logical_and(hi == 0, hj > 0)
                new_arr[hokinds] = np.nan
                bkg_arr[hi_okinds] = 10
                bkg_arr[hj_okinds] = -10
                im.set_array(new_arr)
                im.set_norm(resi_norm)

                im1 = axes[i, j].hexbin(hex_dict[ti]["zs"],
                                        hex_dict[ti]["dens"],
                                        gridsize=20, linewidth=0.2,
                                        yscale="log",
                                        cmap="cmr.guppy", norm=outlier_norm,
                                        extent=extent, alpha=0.2)
                im1.set_array(bkg_arr)
                im1.set_norm(resi_norm)

                # Set up colorbar
                cbar = fig.colorbar(im, cax2, orientation="horizontal")
                cbar.set_label("$\log_{10}(P_{i} / P_{j})$")
                cbar.ax.xaxis.set_ticks_position('top')
                cbar.ax.xaxis.set_label_position('top')
                cbar.ax.xaxis.set_ticks([-2, -1, 0, 1, 2])
                cbar.ax.xaxis.set_ticklabels(
                    ["$\leq-2$", "-1", "0", "1", "$2\leq$"])

    fig.savefig("plots/physics_vary/stellar_birthden_residual.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_birth_denmet_vary(snap):

    # Define redshift bins
    zbins = list(np.arange(5, 12.5, 2.5))
    zbins.append(np.inf)

    # Define EAGLE subgrid  parameters
    parameters = {"f_th,min": 0.3,
                  "f_th,max": 3,
                  "n_Z": 1.0,
                  "n_n": 1.0,
                  "Z_pivot": 0.1 * 0.012,
                  "n_pivot": 0.67}

    star_formation_parameters = {"threshold_Z0": 0.002,
                                 "threshold_n0": 0.1,
                                 "slope": -0.64}

    number_of_bins = 128

    # Constants; these could be put in the parameter file but are
    # rarely changed
    birth_density_bins = np.logspace(-2.9, 6.8, number_of_bins)
    metal_mass_fraction_bins = np.logspace(-5.9, 0, number_of_bins)

    # Now need to make background grid of f_th.
    birth_density_grid, metal_mass_fraction_grid = np.meshgrid(
        0.5 * (birth_density_bins[1:] + birth_density_bins[:-1]),
        0.5 * (metal_mass_fraction_bins[1:] + metal_mass_fraction_bins[:-1]))

    f_th_grid = parameters["f_th,min"] + (parameters["f_th,max"]
                                          - parameters["f_th,min"]) / (
        1.0
        + (metal_mass_fraction_grid /
           parameters["Z_pivot"]) ** parameters["n_Z"]
        * (birth_density_grid / parameters["n_pivot"]) ** (-parameters["n_n"])
    )

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
              "$z_{\mathrm{reion}}=3.0$", "$z_{\mathrm{reion}}=7.5$",
              "$z_{\mathrm{reion}}=14.0$"]

    # Define plot dimensions
    nrows = 3
    ncols = 3

    # Define norm
    norm = LogNorm(vmin=1, vmax=50000)
    resi_norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    outlier_norm = TwoSlopeNorm(vmin=-10.01, vcenter=0, vmax=10)

    # Define hexbin extent
    extent = [-2.9, 6.8, 0, 0.119]

    # Initialise a dictionary to store the hexbins
    hex_dict = {}

    for (ind, t), l in zip(enumerate(types), labels):

        path = ini_path.replace("<type>", t)

        print(path)

        reg_zs, reg_dens = get_nonmaster_evo_data(
            path, snap, y_key="PartType4/BirthDensity")
        reg_zs, reg_mets = get_nonmaster_evo_data(
            path, snap, y_key="PartType4/SmoothedMetallicity")
        birth_a = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                      snap,
                                      "PartType4/StellarFormationTime",
                                      noH=True, physicalUnits=True,
                                      numThreads=8)

        # Compute the birth redshift
        birth_z = (1 / birth_a) - 1

        # Convert density to hydrogen number density
        reg_dens = (reg_dens * 10**10
                    * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value

        hex_dict[t] = {"zs": birth_z, "dens": reg_dens, "mets": reg_mets}

        # Loop over redshift bins
        for zi in range(len(zbins) - 1):

            okinds = np.logical_and(
                np.logical_and(reg_zs >= zbins[zi],
                               reg_zs < zbins[zi + 1]),
                reg_dens > 0)

            im = plt.hexbin(reg_dens[okinds],
                            reg_mets[okinds],
                            gridsize=20, linewidth=0.2,
                            xscale="log",
                            cmap="coolwarm",
                            extent=extent)

            hex_dict[t]["h_%.2f" % zbins[zi]] = im.get_array()

    # Define labels for each
    labels = ["AGNdT9", "REF",
              "InstantFB", "$Z^0$", "$M_\dot=0$",
              "$f_{\mathrm{th, max}}=10$", "$f_{\mathrm{th, max}}=6$",
              "$f_{\mathrm{th, max}}=4$",
              "$z_{\mathrm{reion}}=3.0$", "$z_{\mathrm{reion}}=7.5$",
              "$z_{\mathrm{reion}}=14.0$"]

    plt.close()

    # Loop over redshift bins
    for zi in range(len(zbins) - 1):

        # Set up the plot
        fig = plt.figure(figsize=(len(labels) * 1.0, len(labels) * 1.0))
        gs = gridspec.GridSpec(nrows=len(labels) + 1, ncols=len(labels) + 1,
                               width_ratios=[20, ] * len(labels) + [1, ],
                               height_ratios=[1, ] + [20, ] * len(labels))
        gs.update(wspace=0.0, hspace=0.0)
        axes = np.zeros((len(labels), len(labels)), dtype=object)
        cax1 = fig.add_subplot(gs[:, -1])
        cax2 = fig.add_subplot(gs[0, :])

        # Loop over models and construct corner plot
        for i, ti in enumerate(types):
            for j, tj in enumerate(types):

                # Create axis
                ax = fig.add_subplot(gs[i + 1, j])

                # Include labels
                if j == 0:
                    ax.set_ylabel(r"$Z_{\mathrm{birth}}$")
                if i == len(labels) - 1:
                    ax.set_xlabel(r"$n_{\mathrm{H}} / \mathrm{cm}^{-3}$")

                # Remove unnecessary ticks
                if j > 0:
                    ax.tick_params("y", left=False, right=False,
                                   labelleft=False, labelright=False)
                if i < len(labels) - 1:
                    ax.tick_params("x", top=False, bottom=False,
                                   labeltop=False, labelbottom=False)

                # Set axis limits
                ax.set_ylim(extent[2], extent[3])
                ax.set_xlim(10**extent[0], 10**extent[1])

                # Label axis
                if j == 0:
                    ax.text(-1.0, 0.5, labels[i],
                            transform=ax.transAxes, verticalalignment='center',
                            rotation=90)
                if i == len(labels) - 1:
                    ax.text(0.5, -1.0, labels[j],
                            transform=ax.transAxes, horizontalalignment='center')

                axes[i, j] = ax

                if j == i:

                    zs = hex_dict[ti]["zs"]
                    dens = hex_dict[ti]["dens"]
                    mets = hex_dict[ti]["mets"]

                    okinds = np.logical_and(
                        np.logical_and(zs >= zbins[zi], zs < zbins[zi + 1]),
                        dens > 0)

                    im = axes[i, j].hexbin(dens[okinds],
                                           mets[okinds],
                                           mincnt=0, gridsize=20,
                                           linewidth=0.2, norm=norm,
                                           xscale="log",
                                           cmap="plasma",
                                           extent=extent)

                    # Set up colorbar
                    cbar = fig.colorbar(im, cax1)
                    cbar.set_label("$N$")

                else:

                    zs = hex_dict[ti]["zs"]
                    dens = hex_dict[ti]["dens"]
                    mets = hex_dict[ti]["mets"]
                    okinds = np.logical_and(
                        np.logical_and(
                            zs >= zbins[zi], zs < zbins[zi + 1]),
                        dens > 0)

                    im = axes[i, j].hexbin(dens[okinds], mets[okinds],
                                           gridsize=20, linewidth=0.2,
                                           xscale="log",
                                           cmap="cmr.guppy",
                                           extent=extent)
                    hi = hex_dict[ti]["h_%.2f" % zbins[zi]]
                    hj = hex_dict[tj]["h_%.2f" % zbins[zi]]
                    hokinds = np.logical_and(hi == 0,
                                             hj == 0)
                    new_arr = np.log10((hi / np.sum(hi)) / (hj / np.sum(hj)))
                    bkg_arr = np.full_like(new_arr, np.nan)
                    print(np.min(new_arr), np.max(new_arr))
                    hi_okinds = np.logical_and(hi > 0, hj == 0)
                    hj_okinds = np.logical_and(hi == 0, hj > 0)
                    new_arr[hokinds] = np.nan
                    bkg_arr[hi_okinds] = 10
                    bkg_arr[hj_okinds] = -10
                    im.set_array(new_arr)
                    im.set_norm(resi_norm)

                    im1 = axes[i, j].hexbin(dens[okinds],
                                            mets[okinds],
                                            xscale="log",
                                            gridsize=20, linewidth=0.2,
                                            cmap="cmr.guppy", norm=outlier_norm,
                                            extent=extent, alpha=0.2)
                    im1.set_array(bkg_arr)
                    im1.set_norm(resi_norm)

                    # Set up colorbar
                    cbar = fig.colorbar(im, cax2, orientation="horizontal")
                    cbar.set_label("$\log_{10}(P_{i} / P_{j})$")
                    cbar.ax.xaxis.set_ticks_position('top')
                    cbar.ax.xaxis.set_label_position('top')
                    cbar.ax.xaxis.set_ticks([-2, -1, 0, 1, 2])
                    cbar.ax.xaxis.set_ticklabels(
                        ["$\leq-2$", "-1", "0", "1", "$2\leq$"])

        spath = ("plots/physics_vary/"
                 "stellar_birthdenmet_residual_%.1f-%.1f.png" % (zbins[zi],
                                                                 zbins[zi + 1]))

        fig.savefig(spath.replace(".", "p"),
                    bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    plot_birth_met_evo()
    plot_birth_density_evo()
    plot_birth_met_vary("010_z005p000")
    plot_birth_den_vary("010_z005p000")
    plot_birth_denmet_vary("010_z005p000")

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


def plot_hmr_phys_comp(snap):

    mass_bins = np.logspace(7.5, 11.5, 30)

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

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Log the y axis
    ax.loglog()

    # Loop over the variants
    for t, l, ls in zip(types, labels, linestyles):

        # Get the arrays from the raw data files
        hmr = eagle_io.read_array('SUBFIND', path.replace("<type>", t),
                                  snap,
                                  'Subhalo/HalfMassRad',
                                  noH=True, physicalUnits=True,
                                  numThreads=8)[:, 4] * 1000
        mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/ApertureMeasurements/Mass/030kpc",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)[:, 4] * 10 ** 10

        # Plot median curves
        okinds = mass > 0
        plot_meidan_stat(mass[okinds], hmr[okinds], np.ones(hmr[okinds].size),
                         ax, lab=l, color=None, bins=mass_bins, ls=ls)

    # Label axes
    ax.set_ylabel(r"$R_{1/2}$")
    ax.set_xlabel(r"$M_{\star} / M_\odot$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/stellar_hmr_%s.png" % snap,
                bbox_inches="tight")


def plot_hmr_phys_comp_grid(snap):

    mass_bins = np.logspace(8.0, 13, 30)
    mass_lims = [(10**7.8, 10**11.5), (10**7.8, 10**12.5),
                 (10**7.8, 10**11.2), (10**7.8, 10**12.5)]
    hmr_lims = [(10**0, 10**2), (10**0, 10**2), (10**-0.8, 10**1.3)]

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

    # Define plot grid shape
    nrows = 3
    ncols = 3

    # Set up plot
    fig = plt.figure(figsize=(3.5 * ncols, 3.5 * nrows))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    gs.update(wspace=0.0, hspace=0.0)
    axes = np.empty((nrows, ncols), dtype=object)
    i = 0
    while i < nrows:
        j = 0
        while j < ncols:
            axes[i, j] = fig.add_subplot(gs[i, j])
            axes[i, j].loglog()
            axes[i, j].grid(True)
            if j > 0:
                axes[i, j].tick_params(axis='y', left=False, right=False,
                                       labelleft=False, labelright=False)
            if i < nrows - 1:
                axes[i, j].tick_params(axis='x', top=False, bottom=False,
                                       labeltop=False, labelbottom=False)
            j += 1
        i += 1

    for i in range(axes.shape[0]):

        if i == 0 or i == 1:
            idata = i
        elif i == 2:
            idata = 4

        for j in range(axes.shape[1]):

            if j == 3:
                jdata = "tot"
            elif j == 0 or j == 1:
                jdata = j
            elif j == 2:
                jdata = 4

            # Loop over the variants
            for t, l, ls in zip(types, labels, linestyles):

                print(i, j, t, l)

                # Get the number stars in a galaxy to perform nstar cut
                nparts = eagle_io.read_array('SUBFIND',
                                             path.replace("<type>", t),
                                             snap,
                                             'Subhalo/SubLengthType',
                                             noH=True, physicalUnits=True,
                                             numThreads=8)
                okinds = np.logical_and(nparts[:, 4] > 0, nparts[:, 0] > 0)
                okinds = np.logical_and(okinds, nparts[:, 1] > 0)

                # Get the arrays from the raw data files
                hmr = eagle_io.read_array('SUBFIND', path.replace("<type>", t),
                                          snap,
                                          'Subhalo/HalfMassRad',
                                          noH=True, physicalUnits=True,
                                          numThreads=8)[:, idata] * 1000
                if jdata == "tot":
                    mass_star = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/030kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 4] * 10 ** 10
                    mass_gas = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/030kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 0] * 10 ** 10
                    mass_dm = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/030kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 1] * 10 ** 10
                    mass_bh = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/030kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 5] * 10 ** 10
                    mass = mass_star + mass_dm + mass_gas + mass_bh
                else:
                    mass = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/030kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, jdata] * 10 ** 10

                # Plot median curves
                # okinds = mass > 0
                plot_meidan_stat(mass[okinds], hmr[okinds],
                                 np.ones(hmr[okinds].size),
                                 axes[i, j], lab=l,
                                 color=None, bins=mass_bins, ls=ls)

    # Label axes
    subscripts = ["\mathrm{gas}", "\mathrm{DM}", "\star", "\mathrm{tot}"]
    for ind, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(r"$R_{1/2, %s}$" % subscripts[ind])
    for ind, ax in enumerate(axes[-1, :]):
        ax.set_xlabel(r"$M_{%s} / M_\odot$" % subscripts[ind])

    # Set axis limits
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].set_ylim(hmr_lims[i])
            axes[i, j].set_xlim(mass_lims[j])

    axes[-1, 1].legend(loc='upper center',
                       bbox_to_anchor=(0.5, -0.2),
                       fancybox=True, ncol=6)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/hmr_grid_%s.png" % snap,
                bbox_inches="tight")


def plot_hmr_phys_comp_grid_1kpc(snap):

    mass_bins = np.logspace(6.0, 11, 30)
    mass_lims = [(10**6.0, 10**11), (10**6.0, 10**11),
                 (10**6.0, 10**11), (10**6.0, 10**11)]
    hmr_lims = [10**-0.8, 10**2]

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
              "$z_{r, 0}$", "$z_{r, 7.5}$",
              "$z_{r, 14}$"]
    labels = labels[::-1]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted", "dotted",
                  "dashdot", "dashdot", "dashdot"]
    linestyles = linestyles[::-1]

    # Define plot grid shape
    nrows = 3
    ncols = 4

    # Set up plot
    fig = plt.figure(figsize=(3.5 * ncols, 3.5 * nrows))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    gs.update(wspace=0.0, hspace=0.0)
    axes = np.empty((nrows, ncols), dtype=object)
    i = 0
    while i < nrows:
        j = 0
        while j < ncols:
            axes[i, j] = fig.add_subplot(gs[i, j])
            axes[i, j].loglog()
            if j > 0:
                axes[i, j].tick_params(axis='y', left=False, right=False,
                                       labelleft=False, labelright=False)
            if i < nrows - 1:
                axes[i, j].tick_params(axis='x', top=False, bottom=False,
                                       labeltop=False, labelbottom=False)
            j += 1
        i += 1

    for i in range(axes.shape[0]):

        if i == 0 or i == 1:
            idata = i
        elif i == 2:
            idata = 4

        for j in range(axes.shape[1]):

            if j == 3:
                jdata = "tot"
            elif j == 0 or j == 1:
                jdata = j
            elif j == 2:
                jdata = 4

            # Loop over the variants
            for t, l, ls in zip(types, labels, linestyles):

                print(i, j, t, l)

                # Get the number stars in a galaxy to perform nstar cut
                nparts = eagle_io.read_array('SUBFIND',
                                             path.replace("<type>", t),
                                             snap,
                                             'Subhalo/SubLengthType',
                                             noH=True, physicalUnits=True,
                                             numThreads=8)
                okinds = np.logical_and(nparts[:, 4] > 0, nparts[:, 0] > 0)
                okinds = np.logical_and(okinds, nparts[:, 1] > 0)

                # Get the arrays from the raw data files
                hmr = eagle_io.read_array('SUBFIND', path.replace("<type>", t),
                                          snap,
                                          'Subhalo/HalfMassRad',
                                          noH=True, physicalUnits=True,
                                          numThreads=8)[:, idata] * 1000
                if jdata == "tot":
                    mass_star = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/001kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 4] * 10 ** 10
                    mass_gas = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/001kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 0] * 10 ** 10
                    mass_dm = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/001kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 1] * 10 ** 10
                    mass_bh = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/001kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 5] * 10 ** 10
                    mass = mass_star + mass_dm + mass_gas + mass_bh
                else:
                    mass = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/001kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, jdata] * 10 ** 10

                # Plot median curves
                # okinds = mass > 0
                plot_meidan_stat(mass[okinds], hmr[okinds],
                                 np.ones(hmr[okinds].size),
                                 axes[i, j], lab=l,
                                 color=None, bins=mass_bins, ls=ls)

    # Label axes
    subscripts = ["\mathrm{gas}", "\mathrm{DM}", "\star", "\mathrm{tot}"]
    for ind, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(r"$R_{1/2, %s}$" % subscripts[ind])
    for ind, ax in enumerate(axes[-1, :]):
        ax.set_xlabel(r"$M_{%s} / M_\odot$" % subscripts[ind])

    # Set axis limits
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].set_ylim(hmr_lims)
            axes[i, j].set_xlim(mass_lims[j])

    axes[-1, 1].legend(loc='upper center',
                       bbox_to_anchor=(1.0, -0.2),
                       fancybox=True, ncol=7)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/hmr_grid_1kpc_%s.png" % snap,
                bbox_inches="tight")


def plot_gashmr_phys_comp(snap):

    mass_bins = np.logspace(7.5, 11.5, 30)

    # Define the path
    path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

    # Define physics variations directories
    types = ["G-EAGLE_00", "FLARES_00_REF", "FLARES_00_highFBlim",
             "FLARES_00_medFBlim", "FLARES_00_slightFBlim",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh"]

    # Define labels for each
    labels = ["AGNdT9", "REF", "$f_{\mathrm{th, max}}=10$",
              "$f_{\mathrm{th, max}}=6$", "$f_{\mathrm{th, max}}=4$",
              "InstantFB", "$Z^0$"]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted"]

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Log the y axis
    ax.loglog()

    # Loop over the variants
    for t, l, ls in zip(types, labels, linestyles):

        # Get the arrays from the raw data files
        hmr = eagle_io.read_array('SUBFIND', path.replace("<type>", t),
                                  snap,
                                  'Subhalo/HalfMassRad',
                                  noH=True, physicalUnits=True,
                                  numThreads=8)[:, 0] * 1000
        mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/ApertureMeasurements/Mass/030kpc",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)[:, 4] * 10 ** 10

        # Plot median curves
        okinds = mass > 0
        plot_meidan_stat(mass[okinds], hmr[okinds], np.ones(hmr[okinds].size),
                         ax, lab=l, bins=mass_bins, color=None, ls=ls)

    # Label axes
    ax.set_ylabel(r"$R_{1/2}$")
    ax.set_xlabel(r"$M_{\star} / M_\odot$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/gas_hmr_%s.png" % snap,
                bbox_inches="tight")


def plot_weighted_gas_size_mass_vary(snap):

    # Define redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # What redshift was 100 Myrs ago?
    z_100 = z_at_value(cosmo.age, cosmo.age(z) - (0.1 * u.Gyr),
                       zmin=0, zmax=50)

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
              "$z_{r, 0}$", "$z_{r, 7.5}$",
              "$z_{r, 14}$"]

    # Define plot dimensions
    nrows = 3
    ncols = 3

    # Define norm
    norm = LogNorm(vmin=1, vmax=10)

    # Define hexbin extent
    extent = [8, 11.5, -1.5, 1.5]
    extent1 = [8, 11.5, -3, 2]

    # Set up the plots
    fig = plt.figure(figsize=(nrows * 3.5, ncols * 3.5))
    fig1 = plt.figure(figsize=(nrows * 3.5, ncols * 3.5))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    gs1 = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                            width_ratios=[20, ] * ncols + [1, ])
    gs1.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[-1, -1])
    axes1 = []
    cax1 = fig1.add_subplot(gs1[-1, -1])

    for i in range(nrows):
        for j in range(ncols):

            if i * ncols + j >= len(labels):
                continue

            if labels[i * ncols + j] == "SKIP":
                continue

            # Create axis
            ax = fig.add_subplot(gs[i, j])
            ax1 = fig1.add_subplot(gs1[i, j])

            # Include labels
            if j == 0:
                ax.set_ylabel(r"$R_{1/2} / [\mathrm{pkpc}]$")
                ax1.set_ylabel(r"$R_{gas,1/2} / R_{\star,1/2}$")
            if i == nrows - 1:
                ax.set_xlabel(r"$M_\star / M_\odot$")
                ax1.set_xlabel(r"$M_\star / M_\odot$")

            # Remove unnecessary ticks
            if j > 0:
                ax.tick_params("y", left=False, right=False,
                               labelleft=False, labelright=False)
                ax1.tick_params("y", left=False, right=False,
                                labelleft=False, labelright=False)
            if i < nrows - 1:
                ax.tick_params("x", top=False, bottom=False,
                               labeltop=False, labelbottom=False)
                ax1.tick_params("x", top=False, bottom=False,
                                labeltop=False, labelbottom=False)

            # Set axis limits
            ax.set_ylim(10**extent[2], 10**extent[3])
            ax.set_xlim(10**extent[0], 10**extent[1])
            ax1.set_ylim(10**extent1[2], 10**extent1[3])
            ax1.set_xlim(10**extent1[0], 10**extent1[1])

            # Label axis
            ax.text(0.95, 0.9, labels[i * ncols + j],
                    bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                              alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)
            ax1.text(0.95, 0.9, labels[i * ncols + j],
                     bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                               alpha=0.8),
                     transform=ax1.transAxes, horizontalalignment='right',
                     fontsize=8)

            axes.append(ax)
            axes1.append(ax1)

    for (ind, t), l in zip(enumerate(types), labels):

        path = ini_path.replace("<type>", t)

        print(path)

        mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/ApertureMeasurements/Mass/030kpc",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)[:, 4] * 10 ** 10
        gal_gmass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                        snap,
                                        "Subhalo/ApertureMeasurements/Mass/030kpc",
                                        noH=True, physicalUnits=True,
                                        numThreads=8)[:, 0] * 10 ** 10
        hmrs = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/HalfMassRad",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)[:, 4] * 10 ** 3
        cops = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/CentreOfPotential",
                                   noH=True, physicalUnits=True,
                                   numThreads=8) * 1000
        grps = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/GroupNumber",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)
        subgrps = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                      snap,
                                      "Subhalo/SubGroupNumber",
                                      noH=True, physicalUnits=True,
                                      numThreads=8)
        g_den = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                    snap,
                                    "PartType0/Density",
                                    noH=True, physicalUnits=True,
                                    numThreads=8)
        g_mass = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                     snap,
                                     "PartType0/Mass",
                                     noH=True, physicalUnits=True,
                                     numThreads=8)
        coords = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                     snap,
                                     "PartType0/Coordinates",
                                     noH=True, physicalUnits=True,
                                     numThreads=8) * 1000
        part_grps = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                        snap,
                                        "PartType0/GroupNumber",
                                        noH=True, physicalUnits=True,
                                        numThreads=8)
        part_subgrps = eagle_io.read_array("PARTDATA",
                                           path.replace("<type>", t),
                                           snap,
                                           "PartType0/SubGroupNumber",
                                           noH=True, physicalUnits=True,
                                           numThreads=8)

        # Apply some cuts
        mokinds = np.logical_and(mass > 10**8, gal_gmass > 10**8)
        mass = mass[mokinds]
        cops = cops[mokinds, :]
        grps = grps[mokinds]
        subgrps = subgrps[mokinds]
        hmrs = hmrs[mokinds]

        # Set up array to store sfrs
        w_hmrs = []
        s_hmrs = []
        ms = []

        # Loop over galaxies
        for igal in range(mass.size):

            # Get galaxy data
            m = mass[igal]
            cop = cops[igal, :]
            g = grps[igal]
            sg = subgrps[igal]
            hmr = hmrs[igal]

            # Get this galaxies stars
            gokinds = np.logical_and(part_grps == g, part_subgrps == sg)
            this_coords = coords[gokinds, :] - cop
            this_den = g_den[gokinds]
            this_gmass = g_mass[gokinds]

            # Compute stellar radii
            rs = np.sqrt(this_coords[:, 0] ** 2
                         + this_coords[:, 1] ** 2
                         + this_coords[:, 2] ** 2)

            # Get only particles within the aperture
            rokinds = rs < 30
            rs = rs[rokinds]
            this_den = this_den[rokinds]
            this_gmass = this_gmass[rokinds]

            # Calculate weighted hmr
            weighted_mass = this_gmass * this_den / np.sum(this_den)
            tot = np.sum(weighted_mass)
            half = tot / 2
            sinds = np.argsort(rs)
            rs = rs[sinds]
            weighted_mass = weighted_mass[sinds]
            summed_mass = np.cumsum(weighted_mass)
            g_hmr = rs[np.argmin(np.abs(summed_mass - half))]

            # Compute and store ssfr
            w_hmrs.append(g_hmr)
            ms.append(m)
            s_hmrs.append(hmr)

        # Convert to arrays
        w_hmrs = np.array(w_hmrs)
        ms = np.array(ms)
        s_hmrs = np.array(s_hmrs)

        # Define bins
        bin_edges = np.logspace(extent[0], extent[1], 20)

        okinds = np.logical_and(w_hmrs > 0, s_hmrs > 0)

        im = axes[ind].hexbin(ms[okinds], w_hmrs[okinds], mincnt=1, gridsize=50,
                              xscale="log", yscale="log", linewidth=0.2,
                              cmap="plasma", norm=norm, extent=extent)
        plot_meidan_stat(ms[okinds], w_hmrs[okinds],
                         np.ones(w_hmrs[okinds].size),
                         axes[ind], "", "r", bin_edges)
        im1 = axes1[ind].hexbin(ms[okinds], w_hmrs[okinds] / s_hmrs[okinds],
                                mincnt=1, gridsize=50,
                                xscale="log", yscale="log", linewidth=0.2,
                                cmap="plasma", norm=norm, extent=extent1)
        plot_meidan_stat(ms[okinds], w_hmrs[okinds] / s_hmrs[okinds],
                         np.ones(w_hmrs[okinds].size),
                         axes1[ind], "", "r", bin_edges)

    # Set up colorbar
    cbar = fig.colorbar(im, cax)
    cbar.set_label("$N$")
    cbar1 = fig1.colorbar(im1, cax1)
    cbar1.set_label("$N$")

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/weight_gas_hmr_mass_%s.png" % snap,
                bbox_inches="tight")
    fig1.savefig("plots/physics_vary/weight_gas_hmr_ratio_mass%s.png" % snap,
                 bbox_inches="tight")


def plot_stellar_size_mass_evo_vary():

    # Define binning
    mass_bins = np.logspace(8.0, 13, 30)

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
              "$z_{r, 0}$", "$z_{r, 7.5}$",
              "$z_{r, 14}$"]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted", "dotted",
                  "dashdot", "dashdot", "dashdot"]

    # Define snapshots
    snaps = ["006_z009p000", "007_z008p000", "008_z007p000",
             "009_z006p000", "010_z005p000"]

    # Define plot dimensions
    nrows = 1
    ncols = len(snaps)

    # Define norm
    norm = LogNorm(vmin=1, vmax=10)

    # Define hexbin extent
    extent = [8, 11.5, -1.3, 1.5]

    # Set up the plots
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.5))
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
            ax.set_ylabel(r"$R_{\star,1/2} / [\mathrm{pkpc}]$")
        ax.set_xlabel(r"$M_\star / M_\odot$")

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

    for ax, snap in zip(axes, snaps):

        for (ind, t), l in zip(enumerate(types), labels):

            path = ini_path.replace("<type>", t)

            print(path)

            # Get the arrays from the raw data files
            hmr = eagle_io.read_array('SUBFIND', path.replace("<type>", t),
                                      snap,
                                      'Subhalo/HalfMassRad',
                                      noH=True, physicalUnits=True,
                                      numThreads=8)[:, 4] * 1000
            mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                       snap,
                                       "Subhalo/ApertureMeasurements/Mass/030kpc",
                                       noH=True, physicalUnits=True,
                                       numThreads=8)[:, 4] * 10 ** 10

            okinds = mass > 10 ** 8

            plot_meidan_stat(mass[okinds], hmr[okinds],
                             np.ones(mass[okinds].size),
                             ax, lab=l,
                             color=None, bins=mass_bins,
                             ls=linestyles[ind])

    # Draw legend
    axes[2].legend(loc='upper center',
                   bbox_to_anchor=(0.5, -0.2),
                   fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/sizes/")
    fig.savefig("plots/sizes/stellar_size_mass_evo.png",
                bbox_inches="tight")


def plot_gas_size_mass_evo_vary():

    # Define binning
    mass_bins = np.logspace(8.0, 13, 30)

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
              "$z_{r, 0}$", "$z_{r, 7.5}$",
              "$z_{r, 14}$"]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted", "dotted",
                  "dashdot", "dashdot", "dashdot"]

    # Define snapshots
    snaps = ["006_z009p000", "007_z008p000", "008_z007p000",
             "009_z006p000", "010_z005p000"]

    # Define plot dimensions
    nrows = 1
    ncols = len(snaps)

    # Define norm
    norm = LogNorm(vmin=1, vmax=10)

    # Define hexbin extent
    extent = [8, 11.5, -1.3, 1.5]

    # Set up the plots
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.5))
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
            ax.set_ylabel(r"$R_{\mathrm{gas},1/2} / [\mathrm{pkpc}]$")
        ax.set_xlabel(r"$M_\star / M_\odot$")

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

    for ax, snap in zip(axes, snaps):

        for (ind, t), l in zip(enumerate(types), labels):

            path = ini_path.replace("<type>", t)

            print(path)

            # Get the arrays from the raw data files
            hmr = eagle_io.read_array('SUBFIND', path.replace("<type>", t),
                                      snap,
                                      'Subhalo/HalfMassRad',
                                      noH=True, physicalUnits=True,
                                      numThreads=8)[:, 0] * 1000
            mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                       snap,
                                       "Subhalo/ApertureMeasurements/Mass/030kpc",
                                       noH=True, physicalUnits=True,
                                       numThreads=8)[:, 4] * 10 ** 10

            okinds = mass > 10 ** 8

            plot_meidan_stat(mass[okinds], hmr[okinds],
                             np.ones(mass[okinds].size),
                             ax, lab=l,
                             color=None, bins=mass_bins,
                             ls=linestyles[ind])

    # Draw legend
    axes[2].legend(loc='upper center',
                   bbox_to_anchor=(0.5, -0.2),
                   fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/sizes/")
    fig.savefig("plots/sizes/gas_size_mass_evo.png",
                bbox_inches="tight")


# Run the plotting scripts
if __name__ == "__main__":
    plot_stellar_size_mass_evo_vary()
    plot_gas_size_mass_evo_vary()
    plot_hmr_phys_comp_grid("010_z005p000")

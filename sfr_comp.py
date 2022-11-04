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


def plot_sfr_evo_comp(snap):

    # Define volume
    vol = 4 / 3 * np.pi * 15 ** 3

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
              "$z_{r, 0}$", "$z_{r, 7.5}$",
              "$z_{r, 14}$"]
    labels = labels[::-1]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted", "dotted",
                  "dashdot", "dashdot", "dashdot"]
    linestyles = linestyles[::-1]

    # Define z bins
    flares_age_bins = np.arange(cosmo.age(5).value, cosmo.age(30).value, -0.1)
    flares_z_bins = [5, ]
    for age in flares_age_bins:
        flares_z_bins.append(z_at_value(cosmo.age,
                                        age * u.Gyr,
                                        zmin=0, zmax=50))

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.grid(True)

    # Log the y axis
    ax.loglog()

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.semilogy()
    ax.grid(True)

    # Loop over the variants
    for t, l, ls in zip(types, labels, linestyles):

        # Get data
        zs, ms = get_nonmaster_evo_data(
            path.replace("<type>", t), snap, y_key="PartType4/InitialMass")

        # Loop over reshift bins
        sfrs = []
        plt_zs = []
        for z_low in flares_z_bins[:-1]:

            z_high = z_at_value(cosmo.age, cosmo.age(z_low) - (100 * u.Myr),
                                zmin=0, zmax=50)

            zokinds = np.logical_and(zs < z_high, zs >= z_low)

            sfrs.append(np.sum(ms[zokinds]) * 10**10 / 100)  # M_sun / Myr
            plt_zs.append(z_low)

        sfrs = np.array(sfrs)
        ax.plot(plt_zs, sfrs / vol, label=l, ls=ls)

    ax.set_ylabel(
        r"$\phi / [\mathrm{M}_\odot\mathrm{Myr}^{-1} \mathrm{cMpc}^{-3}]$")
    ax.set_xlabel(r"$z$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/sfr/")
    fig.savefig("plots/sfr/sfr_evo.png",
                bbox_inches="tight")


def plot_ssfr_mass_vary(snap):

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
    extent1 = [-1.5, 1.5, - 2, 1.5]

    # Set up the plots
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.5))
    fig1 = plt.figure(figsize=(ncols * 3.5, nrows * 3.5))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    gs1 = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                            width_ratios=[20, ] * ncols + [1, ])
    gs1.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[1:, -1])
    axes1 = []
    cax1 = fig1.add_subplot(gs1[1:, -1])

    for i in range(nrows):
        for j in range(ncols):

            if i * ncols + j >= len(labels):
                continue

            if labels[i * ncols + j] == "SKIP":
                continue

            # Create axis
            ax = fig.add_subplot(gs[i, j])
            ax1 = fig1.add_subplot(gs1[i, j])
            ax.grid(True)
            ax1.grid(True)

            # Include labels
            if j == 0:
                ax.set_ylabel(r"$\mathrm{sSFR} / [\mathrm{Gyr}^{-1}]$")
                ax1.set_ylabel(r"$\mathrm{sSFR} / [\mathrm{Gyr}^{-1}]$")
            if i == nrows - 1:
                ax.set_xlabel(r"$M_\star / M_\odot$")
                ax1.set_xlabel(r"$R_{1/2} / [\mathrm{pkpc}]$")

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
        birth_a = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                      snap,
                                      "PartType4/StellarFormationTime",
                                      noH=True, physicalUnits=True,
                                      numThreads=8)
        ini_mass = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                       snap,
                                       "PartType4/InitialMass",
                                       noH=True, physicalUnits=True,
                                       numThreads=8) * 10 ** 10
        coords = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                     snap,
                                     "PartType4/Coordinates",
                                     noH=True, physicalUnits=True,
                                     numThreads=8) * 1000
        part_grps = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                        snap,
                                        "PartType4/GroupNumber",
                                        noH=True, physicalUnits=True,
                                        numThreads=8)
        part_subgrps = eagle_io.read_array("PARTDATA",
                                           path.replace("<type>", t),
                                           snap,
                                           "PartType4/SubGroupNumber",
                                           noH=True, physicalUnits=True,
                                           numThreads=8)

        # Apply some cuts
        mokinds = mass > 10**8
        mass = mass[mokinds]
        cops = cops[mokinds, :]
        grps = grps[mokinds]
        subgrps = subgrps[mokinds]
        hmrs = hmrs[mokinds]

        # Compute the birth redshift
        birth_z = (1 / birth_a) - 1

        # Get only particles born since z_100
        okinds = birth_z < z_100
        birth_z = birth_z[okinds]
        ini_mass = ini_mass[okinds]
        coords = coords[okinds, :]
        part_grps = part_grps[okinds]
        part_subgrps = part_subgrps[okinds]

        # Set up array to store sfrs
        ssfrs = []
        ms = []
        plt_hmrs = []

        # Loop over galaxies
        for igal in range(mass.size):

            # Get galaxy data
            m = mass[igal]
            cop = cops[igal, :]
            g = grps[igal]
            sg = subgrps[igal]
            hmr = hmrs[igal]

            # Get this galaxies stars
            sokinds = np.logical_and(part_grps == g, part_subgrps == sg)
            this_coords = coords[sokinds, :] - cop
            this_ini_mass = ini_mass[sokinds]

            # Compute stellar radii
            rs = np.sqrt(this_coords[:, 0] ** 2
                         + this_coords[:, 1] ** 2
                         + this_coords[:, 2] ** 2)

            # Get only particles within the aperture
            rokinds = rs < 30
            this_ini_mass = this_ini_mass[rokinds]

            # Compute and store ssfr
            ssfrs.append(np.sum(this_ini_mass) / 0.1)
            ms.append(m)
            plt_hmrs.append(hmr)

        # Convert to arrays
        sfrs = np.array(ssfrs)
        ms = np.array(ms)
        ssfrs = sfrs / ms
        plt_hmrs = np.array(plt_hmrs)
        okinds = ssfrs > 0

        im = axes[ind].hexbin(ms[okinds], ssfrs[okinds], mincnt=1, gridsize=50,
                              xscale="log", yscale="log", linewidth=0.2,
                              cmap="plasma", norm=norm, extent=extent)
        im1 = axes1[ind].hexbin(plt_hmrs[okinds], ssfrs[okinds], mincnt=1, gridsize=50,
                                xscale="log", yscale="log", linewidth=0.2,
                                cmap="plasma", norm=norm, extent=extent1)

    # Set up colorbar
    cbar = fig.colorbar(im, cax)
    cbar.set_label("$N$")
    cbar1 = fig1.colorbar(im1, cax1)
    cbar1.set_label("$N$")

    # Save figure
    mkdir("plots/sfr/")
    fig.savefig("plots/sfr/sfr_mass_%s.png" % snap,
                bbox_inches="tight")
    fig1.savefig("plots/sfr/sfr_hmr_%s.png" % snap,
                 bbox_inches="tight")


def plot_ssfr_mass_evo_vary():

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
    extent = [8, 11.5, -2, 1.5]

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
        ax.loglog()

        # Include labels
        if j == 0:
            ax.set_ylabel(r"$\mathrm{sSFR} / [\mathrm{Gyr}^{-1}]$")
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

        # Define redshift
        z = float(snap.split("z")[-1].replace("p", "."))

        # What redshift was 100 Myrs ago?
        z_100 = z_at_value(cosmo.age, cosmo.age(z) - (0.1 * u.Gyr),
                           zmin=0, zmax=50)

        for (ind, t), l in zip(enumerate(types), labels):

            path = ini_path.replace("<type>", t)

            print(path)

            mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                       snap,
                                       "Subhalo/ApertureMeasurements/Mass/030kpc",
                                       noH=True, physicalUnits=True,
                                       numThreads=8)[:, 4] * 10 ** 10
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
            birth_a = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                          snap,
                                          "PartType4/StellarFormationTime",
                                          noH=True, physicalUnits=True,
                                          numThreads=8)
            ini_mass = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                           snap,
                                           "PartType4/InitialMass",
                                           noH=True, physicalUnits=True,
                                           numThreads=8) * 10 ** 10
            coords = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                         snap,
                                         "PartType4/Coordinates",
                                         noH=True, physicalUnits=True,
                                         numThreads=8) * 1000
            part_grps = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                            snap,
                                            "PartType4/GroupNumber",
                                            noH=True, physicalUnits=True,
                                            numThreads=8)
            part_subgrps = eagle_io.read_array("PARTDATA",
                                               path.replace("<type>", t),
                                               snap,
                                               "PartType4/SubGroupNumber",
                                               noH=True, physicalUnits=True,
                                               numThreads=8)

            # Apply some cuts
            mokinds = mass > 10**8
            mass = mass[mokinds]
            cops = cops[mokinds, :]
            grps = grps[mokinds]
            subgrps = subgrps[mokinds]
            hmrs = hmrs[mokinds]

            # Compute the birth redshift
            birth_z = (1 / birth_a) - 1

            # Get only particles born since z_100
            okinds = birth_z < z_100
            birth_z = birth_z[okinds]
            ini_mass = ini_mass[okinds]
            coords = coords[okinds, :]
            part_grps = part_grps[okinds]
            part_subgrps = part_subgrps[okinds]

            # Set up array to store sfrs
            ssfrs = []
            ms = []
            plt_hmrs = []

            # Loop over galaxies
            for igal in range(mass.size):

                # Get galaxy data
                m = mass[igal]
                cop = cops[igal, :]
                g = grps[igal]
                sg = subgrps[igal]
                hmr = hmrs[igal]

                # Get this galaxies stars
                sokinds = np.logical_and(part_grps == g, part_subgrps == sg)
                this_coords = coords[sokinds, :] - cop
                this_ini_mass = ini_mass[sokinds]

                # Compute stellar radii
                rs = np.sqrt(this_coords[:, 0] ** 2
                             + this_coords[:, 1] ** 2
                             + this_coords[:, 2] ** 2)

                # Get only particles within the aperture
                rokinds = rs < 30
                this_ini_mass = this_ini_mass[rokinds]

                # Compute and store ssfr
                ssfrs.append(np.sum(this_ini_mass) / 0.1)
                ms.append(m)
                plt_hmrs.append(hmr)

            # Convert to arrays
            sfrs = np.array(ssfrs)
            ms = np.array(ms)
            ssfrs = sfrs / ms
            plt_hmrs = np.array(plt_hmrs)
            okinds = ssfrs > 0

            plot_meidan_stat(ms[okinds], sfrs[okinds],
                             np.ones(sfrs[okinds].size),
                             ax, lab=l,
                             color=None, bins=mass_bins,
                             ls=linestyles[ind])

    # Draw legend
    axes[2].legend(loc='upper center',
                   bbox_to_anchor=(0.5, -0.2),
                   fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/sfr/")
    fig.savefig("plots/sfr/ssfr_mass_evo.png",
                bbox_inches="tight")


def plot_sfr_func_vary():

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
    extent = [8, 11.5, -2, 1.5]

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
        ax.loglog()

        # Include labels
        if j == 0:
            ax.set_ylabel(r"$\mathrm{sSFR} / [\mathrm{Gyr}^{-1}]$")
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

        # Define redshift
        z = float(snap.split("z")[-1].replace("p", "."))

        # What redshift was 100 Myrs ago?
        z_100 = z_at_value(cosmo.age, cosmo.age(z) - (0.1 * u.Gyr),
                           zmin=0, zmax=50)

        for (ind, t), l in zip(enumerate(types), labels):

            path = ini_path.replace("<type>", t)

            print(path)

            mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                       snap,
                                       "Subhalo/ApertureMeasurements/Mass/030kpc",
                                       noH=True, physicalUnits=True,
                                       numThreads=8)[:, 4] * 10 ** 10
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
            birth_a = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                          snap,
                                          "PartType4/StellarFormationTime",
                                          noH=True, physicalUnits=True,
                                          numThreads=8)
            ini_mass = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                           snap,
                                           "PartType4/InitialMass",
                                           noH=True, physicalUnits=True,
                                           numThreads=8) * 10 ** 10
            coords = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                         snap,
                                         "PartType4/Coordinates",
                                         noH=True, physicalUnits=True,
                                         numThreads=8) * 1000
            part_grps = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                            snap,
                                            "PartType4/GroupNumber",
                                            noH=True, physicalUnits=True,
                                            numThreads=8)
            part_subgrps = eagle_io.read_array("PARTDATA",
                                               path.replace("<type>", t),
                                               snap,
                                               "PartType4/SubGroupNumber",
                                               noH=True, physicalUnits=True,
                                               numThreads=8)

            # Apply some cuts
            mokinds = mass > 10**8
            mass = mass[mokinds]
            cops = cops[mokinds, :]
            grps = grps[mokinds]
            subgrps = subgrps[mokinds]
            hmrs = hmrs[mokinds]

            # Compute the birth redshift
            birth_z = (1 / birth_a) - 1

            # Get only particles born since z_100
            okinds = birth_z < z_100
            birth_z = birth_z[okinds]
            ini_mass = ini_mass[okinds]
            coords = coords[okinds, :]
            part_grps = part_grps[okinds]
            part_subgrps = part_subgrps[okinds]

            # Set up array to store sfrs
            ssfrs = []
            ms = []
            plt_hmrs = []

            # Loop over galaxies
            for igal in range(mass.size):

                # Get galaxy data
                m = mass[igal]
                cop = cops[igal, :]
                g = grps[igal]
                sg = subgrps[igal]
                hmr = hmrs[igal]

                # Get this galaxies stars
                sokinds = np.logical_and(part_grps == g, part_subgrps == sg)
                this_coords = coords[sokinds, :] - cop
                this_ini_mass = ini_mass[sokinds]

                # Compute stellar radii
                rs = np.sqrt(this_coords[:, 0] ** 2
                             + this_coords[:, 1] ** 2
                             + this_coords[:, 2] ** 2)

                # Get only particles within the aperture
                rokinds = rs < 30
                this_ini_mass = this_ini_mass[rokinds]

                # Compute and store ssfr
                ssfrs.append(np.sum(this_ini_mass) / 0.1)
                ms.append(m)
                plt_hmrs.append(hmr)

            # Convert to arrays
            sfrs = np.array(ssfrs)
            ms = np.array(ms)
            ssfrs = sfrs / ms
            plt_hmrs = np.array(plt_hmrs)
            okinds = ssfrs > 0

            plot_meidan_stat(ms[okinds], sfrs[okinds],
                             np.ones(sfrs[okinds].size),
                             ax, lab=l,
                             color=None, bins=mass_bins,
                             ls=linestyles[ind])

    # Draw legend
    axes[2].legend(loc='upper center',
                   bbox_to_anchor=(0.5, -0.2),
                   fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/sfr/")
    fig.savefig("plots/sfr/ssfr_mass_evo.png",
                bbox_inches="tight")


if __name__ == "__main__":
    snap = "010_z005p000"
    plot_ssfr_mass_evo_vary()
    plot_sfr_evo_comp(snap)
    plot_ssfr_mass_vary(snap)
